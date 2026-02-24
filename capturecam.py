import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

import cv2

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False
    logging.warning("Picamera2 not found. Camera features will show a placeholder.")

try:
    from serial_handler import STM32_SerialHandler
except ImportError:
    logging.warning("STM32_SerialHandler not found! Using a stub for testing.")
    class STM32_SerialHandler:
        def __init__(self): self.running = False
        def connect(self): return True
        def disconnect(self): pass
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def emergency_brake(self): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Video settings
VIDEO_FPS    = 30
VIDEO_WIDTH  = 640
VIDEO_HEIGHT = 480


class ManualSteeringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BFMC Manual Steering Control & RGB Camera Capture")
        self.root.geometry("1000x740")

        self.handler = STM32_SerialHandler()
        self.is_connected = False

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- CONNECTION ---
        conn_frame = ttk.LabelFrame(left_frame, text="Connection", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)
        self.btn_connect = ttk.Button(conn_frame, text="Connect to Car", command=self.toggle_connection)
        self.btn_connect.pack(fill=tk.X)
        self.lbl_status = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red")
        self.lbl_status.pack(pady=5)

        # --- STEERING ---
        steer_frame = ttk.LabelFrame(left_frame, text="Steering Control", padding="10")
        steer_frame.pack(fill=tk.X, pady=5)
        self.lbl_steer_val = ttk.Label(steer_frame, text="Angle: 0°")
        self.lbl_steer_val.pack()
        self.slider = ttk.Scale(steer_frame, from_=-20, to=20, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.slider.set(0)
        self.slider.pack(fill=tk.X, pady=5)
        self.btn_center = ttk.Button(steer_frame, text="Check Center (0°)", command=self.center_steering)
        self.btn_center.pack(pady=3)

        # --- COLOR CORRECTION ---
        color_frame = ttk.LabelFrame(left_frame, text="Color Correction (Hardware)", padding="10")
        color_frame.pack(fill=tk.X, pady=5)

        ttk.Label(color_frame, text="Increase Red / decrease Blue\nto fix bluish tint:",
                  foreground="gray").pack(anchor=tk.W)

        ttk.Label(color_frame, text="Red Gain (1.0 – 8.0):").pack(anchor=tk.W, pady=(5, 0))
        self.red_gain_var = tk.DoubleVar(value=3.5)
        self.slider_red = ttk.Scale(color_frame, from_=1.0, to=8.0, orient=tk.HORIZONTAL,
                                    variable=self.red_gain_var, command=self.on_gain_change)
        self.slider_red.pack(fill=tk.X)
        self.lbl_red = ttk.Label(color_frame, text="Red: 3.50")
        self.lbl_red.pack(anchor=tk.E)

        ttk.Label(color_frame, text="Blue Gain (1.0 – 8.0):").pack(anchor=tk.W, pady=(5, 0))
        self.blue_gain_var = tk.DoubleVar(value=1.2)
        self.slider_blue = ttk.Scale(color_frame, from_=1.0, to=8.0, orient=tk.HORIZONTAL,
                                     variable=self.blue_gain_var, command=self.on_gain_change)
        self.slider_blue.pack(fill=tk.X)
        self.lbl_blue = ttk.Label(color_frame, text="Blue: 1.20")
        self.lbl_blue.pack(anchor=tk.E)

        btn_row = ttk.Frame(color_frame)
        btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="Reset Gains", command=self.reset_gains).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_row, text="Apply Now",   command=self.on_gain_change).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.apply_wb = True
        self.btn_wb = ttk.Button(color_frame, text="SW White Balance: ON", command=self.toggle_wb)
        self.btn_wb.pack(fill=tk.X, pady=2)

        # --- EMERGENCY ---
        self.btn_stop = ttk.Button(left_frame, text="EMERGENCY STOP", command=self.emergency_stop)
        self.btn_stop.pack(fill=tk.X, pady=10)
        self.btn_stop.configure(state="disabled")

        # --- CAMERA DISPLAY ---
        cam_frame = ttk.LabelFrame(right_frame, text="Live RPi Camera (RGB)", padding="10")
        cam_frame.pack(fill=tk.BOTH, expand=True)
        self.cam_label = tk.Label(cam_frame, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        # --- RECORDING STATUS BAR ---
        self.lbl_rec = ttk.Label(right_frame, text="⏺  NOT RECORDING", foreground="gray")
        self.lbl_rec.pack(pady=(4, 0))

        # --- CAMERA BUTTONS ---
        btn_cam_row = ttk.Frame(right_frame)
        btn_cam_row.pack(fill=tk.X, pady=5)

        self.btn_capture = ttk.Button(btn_cam_row, text="📷  CAPTURE IMAGE", command=self.capture_image)
        self.btn_capture.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_record = ttk.Button(btn_cam_row, text="⏺  START RECORDING", command=self.toggle_recording)
        self.btn_record.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # --- KEYBOARD ---
        self.root.bind("<KeyPress>",   self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.current_speed = 0.0
        self.target_speed  = 0.0
        self.current_steer = 0.0
        self.target_steer  = 0.0
        self.SPEED_STEP = 30.0
        self.STEER_STEP = 3.0
        self.MAX_SPEED  = 200.0
        self.MAX_STEER  = 20.0

        # --- CAMERA STATE ---
        self.picam2       = None
        self.latest_frame = None  # Always stored as RGB uint8 ndarray

        # --- VIDEO RECORDING STATE ---
        self.video_writer    = None   # cv2.VideoWriter instance
        self.is_recording    = False
        self.record_filename = ""
        self.frame_count     = 0

        # --- CAMERA INIT ---
        if _CAM_AVAILABLE:
            self._init_camera()

        self.start_control_loop()
        self.update_camera_feed()

    # =========================================================================
    # CAMERA INITIALISATION
    # =========================================================================
    def _init_camera(self):
        try:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_video_configuration(
                main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "XRGB8888"},
                controls={
                    "AwbEnable":   False,          # Disable auto white balance
                    "ColourGains": (3.5, 1.2),     # (red_gain, blue_gain) – tune with sliders
                    "AeEnable":    True,
                    "Saturation":  1.4,
                    "Sharpness":   1.2,
                }
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            logging.info("Picamera2 started: XRGB8888, AWB OFF, ColourGains=(3.5, 1.2)")
        except Exception as e:
            logging.error(f"Failed to start camera: {e}")
            self.picam2 = None

    # =========================================================================
    # COLOUR GAIN CONTROLS
    # =========================================================================
    def on_gain_change(self, _=None):
        r = round(self.red_gain_var.get(), 2)
        b = round(self.blue_gain_var.get(), 2)
        self.lbl_red.config(text=f"Red: {r:.2f}")
        self.lbl_blue.config(text=f"Blue: {b:.2f}")
        if self.picam2:
            try:
                self.picam2.set_controls({"AwbEnable": False, "ColourGains": (r, b)})
                logging.info(f"Hardware ColourGains → Red={r}, Blue={b}")
            except Exception as e:
                logging.warning(f"Could not set ColourGains: {e}")

    def reset_gains(self):
        self.red_gain_var.set(3.5)
        self.blue_gain_var.set(1.2)
        self.on_gain_change()

    def toggle_wb(self):
        self.apply_wb = not self.apply_wb
        self.btn_wb.config(text=f"SW White Balance: {'ON' if self.apply_wb else 'OFF'}")

    # =========================================================================
    # SOFTWARE WHITE BALANCE (LAB gray-world)
    # =========================================================================
    def fix_white_balance_lab(self, frame_rgb):
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_ch, a_ch, b_ch = cv2.split(lab)
        a_ch -= (np.mean(a_ch) - 128) * (l_ch / 255.0) * 1.2
        b_ch -= (np.mean(b_ch) - 128) * (l_ch / 255.0) * 1.2
        lab = cv2.merge([
            np.clip(l_ch, 0, 255),
            np.clip(a_ch, 0, 255),
            np.clip(b_ch, 0, 255)
        ]).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # =========================================================================
    # CAMERA FEED
    # =========================================================================
    def update_camera_feed(self):
        if self.picam2:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    # XRGB8888 → 4-channel BGRA on RPi; convert to proper RGB
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if self.apply_wb:
                        frame_rgb = self.fix_white_balance_lab(frame_rgb)

                    self.latest_frame = frame_rgb  # Always RGB

                    # --- Write frame to video if recording ---
                    if self.is_recording and self.video_writer is not None:
                        # OpenCV VideoWriter expects BGR
                        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        self.video_writer.write(bgr)
                        self.frame_count += 1
                        # Update recording label with frame count
                        secs = self.frame_count // VIDEO_FPS
                        self.lbl_rec.config(
                            text=f"🔴  RECORDING  {secs // 60:02d}:{secs % 60:02d}  ({self.frame_count} frames)",
                            foreground="red"
                        )

                    # Display
                    img    = Image.fromarray(frame_rgb, 'RGB')
                    imgtk  = ImageTk.PhotoImage(image=img)
                    self.cam_label.imgtk = imgtk
                    self.cam_label.configure(image=imgtk)

            except Exception as e:
                logging.debug(f"Frame drop: {e}")
        else:
            # Mock frame for offline testing
            mock = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            cv2.putText(mock, "NO CAMERA (MOCK RUN)", (90, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.latest_frame = mock
            img   = Image.fromarray(mock)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        self.root.after(30, self.update_camera_feed)

    # =========================================================================
    # IMAGE CAPTURE
    # =========================================================================
    def capture_image(self):
        if self.latest_frame is not None:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Image.fromarray(self.latest_frame).save(filename)
            logging.info(f"Image saved: {filename}")
            messagebox.showinfo("Capture Saved", f"Image saved to:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame available to capture.")

    # =========================================================================
    # VIDEO RECORDING
    # =========================================================================
    def toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.latest_frame is None:
            messagebox.showwarning("Warning", "No camera feed available to record.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.record_filename = f"video_{timestamp}.avi"

        # Use MJPG codec — widely supported, good quality, true colour
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.record_filename,
            fourcc,
            VIDEO_FPS,
            (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Failed to open VideoWriter.\nCheck codec support.")
            self.video_writer = None
            return

        self.is_recording = True
        self.frame_count  = 0
        self.btn_record.config(text="⏹  STOP RECORDING")
        self.lbl_rec.config(text="🔴  RECORDING  00:00  (0 frames)", foreground="red")
        logging.info(f"Recording started → {self.record_filename}")

    def _stop_recording(self):
        self.is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.btn_record.config(text="⏺  START RECORDING")
        self.lbl_rec.config(
            text=f"✅  Saved: {self.record_filename}  ({self.frame_count} frames)",
            foreground="green"
        )
        logging.info(f"Recording stopped → {self.record_filename} ({self.frame_count} frames)")
        messagebox.showinfo("Recording Saved",
                            f"Video saved to:\n{self.record_filename}\n"
                            f"Total frames: {self.frame_count}")

    # =========================================================================
    # CONTROL LOOP
    # =========================================================================
    def start_control_loop(self):
        self.update_control()
        self.root.after(50, self.start_control_loop)

    def update_control(self):
        if not self.is_connected:
            return

        self.target_speed = self.MAX_SPEED if self.keys['Up'] else (-self.MAX_SPEED if self.keys['Down'] else 0.0)
        self.target_steer = -self.MAX_STEER if self.keys['Left'] else (self.MAX_STEER if self.keys['Right'] else 0.0)

        self.current_speed = self.smooth_move(self.current_speed, self.target_speed, self.SPEED_STEP)
        self.current_steer = self.smooth_move(self.current_steer, self.target_steer, self.STEER_STEP)

        if abs(self.current_speed) > 1 or abs(self.target_speed) > 1:
            self.handler.set_speed(int(self.current_speed))
        else:
            if self.current_speed != 0:
                self.handler.set_speed(0)
                self.current_speed = 0

        if abs(self.current_steer) > 0.5 or abs(self.target_steer) > 0.5:
            self.handler.set_steering(self.current_steer)
            self.slider.set(self.current_steer)
        else:
            if self.current_steer != 0:
                self.handler.set_steering(0)
                self.current_steer = 0
                self.slider.set(0)

    def smooth_move(self, current, target, step):
        if current < target:   return min(current + step, target)
        elif current > target: return max(current - step, target)
        return target

    def on_key_press(self, event):
        if event.keysym in self.keys: self.keys[event.keysym] = True

    def on_key_release(self, event):
        if event.keysym in self.keys: self.keys[event.keysym] = False

    # =========================================================================
    # CONNECTION
    # =========================================================================
    def toggle_connection(self):
        if not self.is_connected:
            self.lbl_status.config(text="Status: Connecting...", foreground="orange")
            self.root.update()
            threading.Thread(target=self._connect_thread).start()
        else:
            self.handler.disconnect()
            self.is_connected = False
            self.update_ui_state(False)

    def _connect_thread(self):
        success = self.handler.connect()
        self.is_connected = success
        self.root.after(0, lambda: self.update_ui_state(success, error=not success))

    def update_ui_state(self, connected, error=False):
        if connected:
            self.lbl_status.config(text="Status: Connected", foreground="green")
            self.btn_connect.config(text="Disconnect")
            self.btn_stop.config(state="normal")
            self.slider.config(state="normal")
        else:
            if error:
                self.lbl_status.config(text="Status: Connection Failed", foreground="red")
                messagebox.showerror("Error", "Could not connect to STM32 board.")
            else:
                self.lbl_status.config(text="Status: Disconnected", foreground="red")
            self.btn_connect.config(text="Connect to Car")
            self.btn_stop.config(state="disabled")

    def on_slider_change(self, val):
        angle = float(val)
        self.lbl_steer_val.config(text=f"Angle: {angle:.1f}°")
        if self.is_connected:
            self.handler.set_steering(angle)

    def center_steering(self):
        self.slider.set(0)
        self.on_slider_change(0)

    def emergency_stop(self):
        if self.is_connected:
            self.handler.emergency_brake()
            self.slider.set(0)
            messagebox.showwarning("STOP", "Emergency Brake Activated!")

    def check_status(self):
        if self.is_connected and not getattr(self.handler, "running", True):
            self.is_connected = False
            self.update_ui_state(False, error=True)
        self.root.after(1000, self.check_status)

    def on_close(self):
        # Stop recording cleanly before exit
        if self.is_recording:
            self._stop_recording()
        if self.is_connected:
            self.handler.disconnect()
        if self.picam2:
            self.picam2.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = ManualSteeringGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()