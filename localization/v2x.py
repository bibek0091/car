"""
v2x.py — BFMC V2X (Vehicle-to-Everything) Communication Layer
==============================================================
Fix-2: Implements the competition-required V2X communication protocol.

Four background daemon threads:
  V2XPositionReceiver  — listens for infrastructure location corrections (UDP)
  V2XStatusReporter    — POSTs car position + state to server every 2.5 s (HTTP)
  V2XObstacleReporter  — POSTs detected obstacles once per new detection (HTTP)
  V2XTrafficLightSub   — subscribes to traffic light state via UDP (1 Hz)

V2XManager wraps all four. Instantiate in Orchestrator.__init__() and call
v2x.start() / v2x.stop().

Placeholder IPs/ports — fill in from the competition infrastructure sheet:
  SERVER_IP   = "192.168.1.1"   # Bosch BFMC server
  SERVER_PORT = 8080
  UDP_LISTEN_PORT = 5000        # incoming infrastructure messages
  TL_LISTEN_PORT  = 5001        # traffic-light subscription updates
"""

import json
import logging
import queue
import socket
import threading
import time
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

# ─── Configuration (replace with actual competition values) ───────────────────
SERVER_IP        = "192.168.1.1"    # ⚠ set on-site
SERVER_PORT      = 8080
UDP_LISTEN_IP    = "0.0.0.0"
UDP_LISTEN_PORT  = 5000
TL_LISTEN_PORT   = 5001
REPORT_INTERVAL  = 2.5             # seconds between status POSTs


# ─── V2XPositionReceiver ──────────────────────────────────────────────────────

class V2XPositionReceiver(threading.Thread):
    """
    Listens for UDP position-correction packets from infrastructure.
    Packets are JSON: {"x": <float>, "y": <float>, "heading": <float>}
    Puts them into a queue the localizer can consume.
    """
    def __init__(self, correction_q: queue.Queue, listen_port: int = UDP_LISTEN_PORT):
        super().__init__(name="V2X-PosRx", daemon=True)
        self._q    = correction_q
        self._port = listen_port
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind((UDP_LISTEN_IP, self._port))
            log.info("V2X-PosRx: listening on UDP :%d", self._port)
        except OSError as e:
            log.warning("V2X-PosRx: bind failed (%s) — disabled", e)
            return

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                if "x" in msg and "y" in msg:
                    # Discard if queue is full (always keep latest)
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    self._q.put_nowait(msg)
                    log.debug("V2X-PosRx: correction x=%.2f y=%.2f", msg["x"], msg["y"])
            except (socket.timeout, json.JSONDecodeError):
                pass
            except Exception as e:
                log.warning("V2X-PosRx: error %s", e)

        sock.close()


# ─── V2XStatusReporter ────────────────────────────────────────────────────────

class V2XStatusReporter(threading.Thread):
    """
    POSTs car position, heading, and state to the competition server
    at REPORT_INTERVAL seconds. Reads from a shared state dict (thread-safe copy).
    """
    def __init__(self, state_fn, interval: float = REPORT_INTERVAL):
        """
        state_fn: callable returning dict with keys:
            x, y, heading, speed_ms, state, timestamp
        """
        super().__init__(name="V2X-Reporter", daemon=True)
        self._state_fn = state_fn
        self._interval = interval
        self._stop     = threading.Event()
        self._url      = f"http://{SERVER_IP}:{SERVER_PORT}/api/car/status"

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("V2X-Reporter: posting to %s every %.1fs", self._url, self._interval)
        while not self._stop.is_set():
            try:
                payload = json.dumps(self._state_fn()).encode()
                req = urllib.request.Request(
                    self._url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status not in (200, 201, 204):
                        log.warning("V2X-Reporter: server returned %d", resp.status)
            except urllib.error.URLError as e:
                log.debug("V2X-Reporter: POST failed (%s)", e.reason)
            except Exception as e:
                log.debug("V2X-Reporter: unexpected error %s", e)

            # Wait until next report, but wake up immediately on stop
            self._stop.wait(timeout=self._interval)


# ─── V2XObstacleReporter ─────────────────────────────────────────────────────

class V2XObstacleReporter(threading.Thread):
    """
    One-shot POSTs detected obstacles to the competition server.
    Obstacle packets arrive via an internal queue (put by the traffic engine).
    Packet format: {"x": float, "y": float, "label": str, "confidence": float}
    """
    def __init__(self):
        super().__init__(name="V2X-ObsRpt", daemon=True)
        self._q    = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._url  = f"http://{SERVER_IP}:{SERVER_PORT}/api/obstacle"

    def report(self, x: float, y: float, label: str, conf: float = 1.0):
        """Non-blocking — drops if queue is full."""
        try:
            self._q.put_nowait({"x": x, "y": y, "label": label, "confidence": conf,
                                "timestamp": time.time()})
        except queue.Full:
            pass  # drop silently — old obstacle reports are stale anyway

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)  # wake run() loop
        except queue.Full:
            pass

    def run(self):
        log.info("V2X-ObsRpt: posting to %s", self._url)
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
                if item is None:
                    break
                payload = json.dumps(item).encode()
                req = urllib.request.Request(
                    self._url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=2.0):
                    pass
                log.debug("V2X-ObsRpt: reported %s at (%.2f, %.2f)",
                          item["label"], item["x"], item["y"])
            except queue.Empty:
                pass
            except urllib.error.URLError as e:
                log.debug("V2X-ObsRpt: POST failed (%s)", e.reason)
            except Exception as e:
                log.debug("V2X-ObsRpt: error %s", e)


# ─── V2XTrafficLightSubscriber ────────────────────────────────────────────────

class V2XTrafficLightSubscriber(threading.Thread):
    """
    Subscribes to infrastructure traffic-light state updates (UDP, ~1 Hz).
    Packet: {"node_id": str, "state": "RED"|"GREEN"|"YELLOW", "phase_s": float}
    Updates a shared dict: tl_states[node_id] = {"state": ..., "ts": ...}
    """
    def __init__(self, tl_states: dict, listen_port: int = TL_LISTEN_PORT):
        super().__init__(name="V2X-TLSub", daemon=True)
        self._tl_states = tl_states   # shared dict, caller must read only
        self._port      = listen_port
        self._stop      = threading.Event()
        self._lock      = threading.Lock()

    def get_state(self, node_id: str):
        """Thread-safe read of a traffic light state. Returns None if unknown."""
        with self._lock:
            entry = self._tl_states.get(node_id)
        if entry is None:
            return None
        # Stale after 5 s
        if time.time() - entry["ts"] > 5.0:
            return None
        return entry["state"]

    def stop(self):
        self._stop.set()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind((UDP_LISTEN_IP, self._port))
            log.info("V2X-TLSub: listening on UDP :%d", self._port)
        except OSError as e:
            log.warning("V2X-TLSub: bind failed (%s) — disabled", e)
            return

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                nid   = msg.get("node_id")
                state = msg.get("state", "").upper()
                if nid and state in ("RED", "GREEN", "YELLOW"):
                    with self._lock:
                        self._tl_states[nid] = {"state": state, "ts": time.time()}
                    log.debug("V2X-TLSub: node %s → %s", nid, state)
            except (socket.timeout, json.JSONDecodeError):
                pass
            except Exception as e:
                log.warning("V2X-TLSub: error %s", e)

        sock.close()


# ─── V2XManager ───────────────────────────────────────────────────────────────

class V2XManager:
    """
    Facade that owns and manages all four V2X threads.

    Usage in Orchestrator.__init__():
        self.v2x = V2XManager(state_fn=self._v2x_state_snapshot)
        self.v2x.start()

    The state_fn must return a dict with at least: x, y, heading, speed_ms, state.
    Access V2X data from orchestrator:
        self.v2x.correction_q   — queue of position corrections
        self.v2x.obstacle_rpt   — call .report(x, y, label, conf)
        self.v2x.tl_sub.get_state(node_id) — current TL state
    """
    def __init__(self, state_fn=None):
        self.correction_q  = queue.Queue(maxsize=1)
        self._tl_states    = {}
        self._state_fn     = state_fn or (lambda: {})

        self._pos_rx       = V2XPositionReceiver(self.correction_q)
        self._reporter     = V2XStatusReporter(self._state_fn)
        self.obstacle_rpt  = V2XObstacleReporter()
        self.tl_sub        = V2XTrafficLightSubscriber(self._tl_states)

        self._threads = [self._pos_rx, self._reporter,
                         self.obstacle_rpt, self.tl_sub]

    def start(self):
        for t in self._threads:
            if not t.is_alive():
                t.start()
        log.info("V2XManager: all 4 threads started")

    def stop(self):
        for t in self._threads:
            t.stop()
        for t in self._threads:
            t.join(timeout=2.0)
        log.info("V2XManager: stopped")

    def drain_corrections(self):
        """Drain all queued position corrections. Returns list of dicts."""
        out = []
        while True:
            try:
                out.append(self.correction_q.get_nowait())
            except queue.Empty:
                break
        return out
