"""
sign_map.py — BFMC Road Sign Map
=================================
Persistent data model for placing named road signs at known map coordinates.

Usage
-----
  sign_map = SignMap("sign_map.json")
  sign_map.add_sign("stop", x_m=4.5, y_m=3.2)
  nearby = sign_map.get_nearby(car_x, car_y, radius_m=4.0)
  matched = sign_map.match_detection("stop", car_x, car_y)
  img = sign_map.draw_on_image(bgr_img, MAP_W_M, MAP_H_M)
"""

import json
import math
import os
import uuid
import logging
import numpy as np
import cv2

log = logging.getLogger(__name__)

# ── Sign type catalogue ────────────────────────────────────────────────────────
SIGN_TYPES = [
    "stop",
    "parking",
    "crosswalk",
    "priority",
    "highway-entry",
    "highway-exit",
    "no-entry",
    "roundabout",
    "speed-limit",
    "traffic-light",
]

# BGR colours and single-char glyphs for map overlay rendering
_SIGN_STYLE: dict = {
    "stop":          ((0,   0, 220), "S"),
    "parking":       ((200, 90,   0), "P"),
    "crosswalk":     ((0, 200, 200), "X"),
    "priority":      ((50, 200,  50), "!"),
    "highway-entry": ((20, 180,  20), "H"),
    "highway-exit":  ((40, 100, 200), "h"),
    "no-entry":      ((0,   0, 200), "N"),
    "roundabout":    ((200, 50, 200), "O"),
    "speed-limit":   ((180,180,   0), "L"),
    "traffic-light": ((0, 200, 100), "T"),
}
_DEFAULT_STYLE = ((160, 160, 160), "?")


class SignMap:
    """
    Stores road signs at known map positions.

    Each sign is a dict:
      { 'id': str, 'type': str, 'x_m': float, 'y_m': float,
        'radius_m': float, 'label': str }

    Signs are persisted to a JSON file automatically on every add/remove.
    """

    DEFAULT_RADIUS_M = 1.5   # detection activation radius

    def __init__(self, filepath: str = "sign_map.json"):
        self.filepath = filepath
        self.signs: list = []
        self._load()
        log.info("SignMap: loaded %d signs from %s", len(self.signs), filepath)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_sign(self, sign_type: str, x_m: float, y_m: float,
                 radius_m: float = None, label: str = "") -> dict:
        """Add a sign and persist. Returns the new sign dict."""
        if sign_type not in SIGN_TYPES:
            log.warning("SignMap: unknown sign type '%s', adding anyway", sign_type)
        entry = {
            "id":       str(uuid.uuid4())[:8],
            "type":     sign_type,
            "x_m":      round(float(x_m), 3),
            "y_m":      round(float(y_m), 3),
            "radius_m": round(float(radius_m or self.DEFAULT_RADIUS_M), 2),
            "label":    label or sign_type.replace("-", " ").upper(),
        }
        self.signs.append(entry)
        self.save()
        log.info("SignMap: added %s @ (%.2f, %.2f)", sign_type, x_m, y_m)
        return entry

    def remove_by_id(self, sign_id: str) -> bool:
        """Remove a sign by its ID. Returns True if found."""
        before = len(self.signs)
        self.signs = [s for s in self.signs if s["id"] != sign_id]
        if len(self.signs) < before:
            self.save()
            return True
        return False

    def remove_last(self) -> bool:
        """Remove the most recently added sign."""
        if self.signs:
            removed = self.signs.pop()
            self.save()
            log.info("SignMap: removed last sign %s", removed["id"])
            return True
        return False

    def remove_nearest(self, x_m: float, y_m: float,
                       max_dist_m: float = 1.0) -> bool:
        """Remove the closest sign within max_dist_m. Returns True if removed."""
        candidates = self._with_dist(x_m, y_m)
        if not candidates:
            return False
        nearest = min(candidates, key=lambda t: t[1])
        if nearest[1] <= max_dist_m:
            return self.remove_by_id(nearest[0]["id"])
        return False

    def clear(self):
        """Remove all signs."""
        self.signs.clear()
        self.save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_nearby(self, x_m: float, y_m: float,
                   radius_m: float = 4.0) -> list:
        """
        Returns signs whose own radius overlaps the query circle.
        Each result dict has an extra 'dist' key (metres to sign centre).
        """
        result = []
        for s, d in self._with_dist(x_m, y_m):
            if d <= max(radius_m, s["radius_m"]):
                entry = dict(s); entry["dist"] = round(d, 3)
                result.append(entry)
        return sorted(result, key=lambda e: e["dist"])

    def match_detection(self, detected_label: str, x_m: float, y_m: float,
                        radius_m: float = 3.0) -> dict | None:
        """
        Match a YOLO-detected label to the nearest placed sign of that type
        within radius_m.  Returns the sign dict (with 'dist') or None.

        Matching is fuzzy: detected_label is checked as a substring of sign type
        and vice-versa to handle partial class names (e.g. 'stop-sign' → 'stop').
        """
        label_lower = detected_label.lower().replace("_", "-").replace(" ", "-")
        best, best_dist = None, float("inf")
        for s, d in self._with_dist(x_m, y_m):
            if d > max(radius_m, s["radius_m"]):
                continue
            stype = s["type"].lower()
            if label_lower in stype or stype in label_lower:
                if d < best_dist:
                    best, best_dist = s, d
        if best:
            result = dict(best); result["dist"] = round(best_dist, 3)
            return result
        return None

    # ── Render ────────────────────────────────────────────────────────────────

    def draw_on_image(self, img: np.ndarray,
                      map_w_m: float, map_h_m: float) -> np.ndarray:
        """
        Draw all placed signs on `img` (BGR) as coloured circles + labels.
        img is assumed to be (h, w, 3).  Modifies and returns img.
        """
        h, w = img.shape[:2]

        for s in self.signs:
            px = int(s["x_m"] / map_w_m * w)
            py = int(s["y_m"] / map_h_m * h)
            color, glyph = _SIGN_STYLE.get(s["type"], _DEFAULT_STYLE)

            # Outer ring
            cv2.circle(img, (px, py), 11, (30, 30, 30), -1)
            cv2.circle(img, (px, py), 10, color, 2)
            # Glyph
            cv2.putText(img, glyph, (px - 5, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
            # Label below
            label_short = s["type"][:9]
            cv2.putText(img, label_short, (px - 22, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1,
                        cv2.LINE_AA)
        return img

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({"signs": self.signs}, f, indent=2)
        except OSError as e:
            log.error("SignMap: save failed: %s", e)

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            self.signs = data.get("signs", [])
        except (json.JSONDecodeError, OSError) as e:
            log.warning("SignMap: load failed (%s) — starting empty", e)
            self.signs = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dist(self, s: dict, x_m: float, y_m: float) -> float:
        return math.hypot(s["x_m"] - x_m, s["y_m"] - y_m)

    def _with_dist(self, x_m: float, y_m: float) -> list:
        return [(s, self._dist(s, x_m, y_m)) for s in self.signs]

    def __len__(self) -> int:
        return len(self.signs)
