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


class ManualSteeringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BFMC Manual Steering Control & RGB Camera Capture")
        self.root.geometry("1000x700")

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

        # Info label
        ttk.Label(color_frame, text="Increase Red / decrease Blue\nto fix bluish tint:",
                  foreground="gray").pack(anchor=tk.W)

        # Red gain
        ttk.Label(color_frame, text="Red Gain (1.0 – 8.0):").pack(anchor=tk.W, pady=(5, 0))
        self.red_gain_var = tk.DoubleVar(value=3.5)
        self.slider_red = ttk.Scale(color_frame, from_=1.0, to=8.0, orient=tk.HORIZONTAL,
                                    variable=self.red_gain_var, command=self.on_gain_change)
        self.slider_red.pack(fill=tk.X)
        self.lbl_red = ttk.Label(color_frame, text="Red: 3.50")
        self.lbl_red.pack(anchor=tk.E)

        # Blue gain
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
        ttk.Button(btn_row, text="Apply Now", command=self.on_gain_change).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Software WB toggle (extra fallback layer)
        self.apply_wb = True
        self.btn_wb = ttk.Button(color_frame, text="SW White Balance: ON", command=self.toggle_wb)
        self.btn_wb.pack(fill=tk.X, pady=2)

        # --- EMERGENCY ---
        self.btn_stop = ttk.Button(left_frame, text="EMERGENCY STOP", command=self.emergency_stop)
        self.btn_stop.pack(fill=tk.X, pady=15)
        self.btn_stop.configure(state="disabled")

        # --- CAMERA DISPLAY ---
        cam_frame = ttk.LabelFrame(right_frame, text="Live RPi Camera (RGB)", padding="10")
        cam_frame.pack(fill=tk.BOTH, expand=True)
        self.cam_label = tk.Label(cam_frame, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        self.btn_capture = ttk.Button(right_frame, text="CAPTURE RGB IMAGE", command=self.capture_image)
        self.btn_capture.pack(fill=tk.X, pady=5)

        # --- KEYBOARD ---
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.current_steer = 0.0
        self.target_steer = 0.0
        self.SPEED_STEP = 30.0
        self.STEER_STEP = 3.0
        self.MAX_SPEED = 200.0
        self.MAX_STEER = 20.0

        self.picam2 = None
        self.latest_frame = None

        # --- CAMERA INIT ---
        if _CAM_AVAILABLE:
            self._init_camera()

        self.start_control_loop()
        self.update_camera_feed()

    # -------------------------------------------------------------------------
    # CAMERA INITIALISATION
    # -------------------------------------------------------------------------
    def _init_camera(self):
        try:
            self.picam2 = Picamera2()

            # XRGB8888 is the most stable 4-channel format on RPi 5
            cfg = self.picam2.create_video_configuration(
                main={"size": (640, 480), "format": "XRGB8888"},
                controls={
                    # ---- KEY FIX: disable auto white balance ----
                    "AwbEnable": False,
                    # Manual colour gains: (red_gain, blue_gain)
                    # High red, low blue removes the typical blue cast.
                    # Use the sliders in the GUI to fine-tune live.
                    "ColourGains": (3.5, 1.2),
                    # Keep auto-exposure on for correct brightness
                    "AeEnable": True,
                    # Slight saturation boost so colours look vivid after WB fix
                    "Saturation": 1.4,
                    "Sharpness": 1.2,
                }
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            logging.info("Picamera2 started: XRGB8888, AWB OFF, ColourGains=(3.5, 1.2)")
        except Exception as e:
            logging.error(f"Failed to start camera: {e}")
            self.picam2 = None

    # -------------------------------------------------------------------------
    # COLOUR GAIN CONTROLS
    # -------------------------------------------------------------------------
    def on_gain_change(self, _=None):
        r = round(self.red_gain_var.get(), 2)
        b = round(self.blue_gain_var.get(), 2)
        self.lbl_red.config(text=f"Red: {r:.2f}")
        self.lbl_blue.config(text=f"Blue: {b:.2f}")
        if self.picam2:
            try:
                self.picam2.set_controls({
                    "AwbEnable": False,
                    "ColourGains": (r, b)
                })
                logging.info(f"Hardware ColourGains set to Red={r}, Blue={b}")
            except Exception as e:
                logging.warning(f"Could not set ColourGains: {e}")

    def reset_gains(self):
        self.red_gain_var.set(3.5)
        self.blue_gain_var.set(1.2)
        self.on_gain_change()

    def toggle_wb(self):
        self.apply_wb = not self.apply_wb
        state = "ON" if self.apply_wb else "OFF"
        self.btn_wb.config(text=f"SW White Balance: {state}")

    # -------------------------------------------------------------------------
    # SOFTWARE WHITE BALANCE (LAB gray-world fallback)
    # -------------------------------------------------------------------------
    def fix_white_balance_lab(self, frame_rgb):
        """
        Removes remaining color cast using LAB color space.
        Shifts A and B channels toward neutral 128.
        """
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_ch, a_ch, b_ch = cv2.split(lab)
        avg_a = np.mean(a_ch)
        avg_b = np.mean(b_ch)
        a_ch -= (avg_a - 128) * (l_ch / 255.0) * 1.2
        b_ch -= (avg_b - 128) * (l_ch / 255.0) * 1.2
        lab = cv2.merge([
            np.clip(l_ch, 0, 255),
            np.clip(a_ch, 0, 255),
            np.clip(b_ch, 0, 255)
        ]).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # -------------------------------------------------------------------------
    # CAMERA FEED
    # -------------------------------------------------------------------------
    def update_camera_feed(self):
        if self.picam2:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    # XRGB8888 = 4 channels (BGRA order from libcamera on RPi)
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        # Fallback: treat as BGR
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Software WB as secondary correction layer
                    if self.apply_wb:
                        frame_rgb = self.fix_white_balance_lab(frame_rgb)

                    self.latest_frame = frame_rgb
                    img = Image.fromarray(frame_rgb, 'RGB')
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.cam_label.imgtk = imgtk
                    self.cam_label.configure(image=imgtk)
            except Exception as e:
                logging.debug(f"Frame drop: {e}")
        else:
            mock = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock, "NO CAMERA (MOCK RUN)", (90, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.latest_frame = mock
            img = Image.fromarray(mock)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        self.root.after(30, self.update_camera_feed)

    def capture_image(self):
        if self.latest_frame is not None:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img = Image.fromarray(self.latest_frame)
            img.save(filename)
            logging.info(f"Image saved: {filename}")
            messagebox.showinfo("Capture Saved", f"Image saved to:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame available to capture.")

    # -------------------------------------------------------------------------
    # CONTROL LOOP
    # -------------------------------------------------------------------------
    def start_control_loop(self):
        self.update_control()
        self.root.after(50, self.start_control_loop)

    def update_control(self):
        if not self.is_connected:
            return

        if self.keys['Up']:
            self.target_speed = self.MAX_SPEED
        elif self.keys['Down']:
            self.target_speed = -self.MAX_SPEED
        else:
            self.target_speed = 0.0

        if self.keys['Left']:
            self.target_steer = -self.MAX_STEER
        elif self.keys['Right']:
            self.target_steer = self.MAX_STEER
        else:
            self.target_steer = 0.0

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
        if current < target:
            return min(current + step, target)
        elif current > target:
            return max(current - step, target)
        return target

    def on_key_press(self, event):
        if event.keysym in self.keys:
            self.keys[event.keysym] = True

    def on_key_release(self, event):
        if event.keysym in self.keys:
            self.keys[event.keysym] = False

    # -------------------------------------------------------------------------
    # CONNECTION
    # -------------------------------------------------------------------------
    def toggle_connection(self):
        if not self.is_connected:
            self.lbl_status.config(text="Status: Connecting...", foreground="orange")
            self.root.update()
            t = threading.Thread(target=self._connect_thread)
            t.start()
        else:
            self.handler.disconnect()
            self.is_connected = False
            self.update_ui_state(False)

    def _connect_thread(self):
        success = self.handler.connect()
        if success:
            self.is_connected = True
            self.root.after(0, lambda: self.update_ui_state(True))
        else:
            self.is_connected = False
            self.root.after(0, lambda: self.update_ui_state(False, error=True))

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
        if self.is_connected:
            if not getattr(self.handler, "running", True):
                self.is_connected = False
                self.update_ui_state(False, error=True)
        self.root.after(1000, self.check_status)

    def on_close(self):
        if self.is_connected:
            self.handler.disconnect()
        if self.picam2:
            self.picam2.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ManualSteeringGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()