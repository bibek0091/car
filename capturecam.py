import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import time
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

# Attempt to load picamera2; stub if on Windows/offline
import cv2
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False
    logging.warning("Picamera2 not found. Camera features will show a placeholder.")

# Attempt to load the user's serial handler
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ManualSteeringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BFMC Manual Steering Control & RGB Camera Capture")
        self.root.geometry("1000x640")
        
        self.handler = STM32_SerialHandler()
        self.is_connected = False
        
        # Styles
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # Main Frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ----------------- LAYOUT SPLIT -----------------
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ----------------- CONNECTION SECTION -----------------
        conn_frame = ttk.LabelFrame(left_frame, text="Connection", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_connect = ttk.Button(conn_frame, text="Connect to Car", command=self.toggle_connection)
        self.btn_connect.pack(fill=tk.X)
        
        self.lbl_status = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red")
        self.lbl_status.pack(pady=5)

        # ----------------- STEERING SECTION -----------------
        steer_frame = ttk.LabelFrame(left_frame, text="Steering Control", padding="10")
        steer_frame.pack(fill=tk.X, pady=10)
        
        self.lbl_steer_val = ttk.Label(steer_frame, text="Angle: 0°")
        self.lbl_steer_val.pack()
        
        # Slider from -20 to +20
        self.slider = ttk.Scale(
            steer_frame, 
            from_=-20, 
            to=20, 
            orient=tk.HORIZONTAL, 
            command=self.on_slider_change
        )
        self.slider.set(0)
        self.slider.pack(fill=tk.X, pady=5)
        
        # Center Button
        self.btn_center = ttk.Button(steer_frame, text="Check Center (0°)", command=self.center_steering)
        self.btn_center.pack(pady=5)

        # ----------------- EMERGENCY SECTION -----------------
        self.btn_stop = ttk.Button(left_frame, text="EMERGENCY STOP", command=self.emergency_stop)
        self.btn_stop.pack(fill=tk.X, pady=20)
        self.btn_stop.configure(state="disabled")

        # ----------------- CAMERA SECTION -----------------
        cam_frame = ttk.LabelFrame(right_frame, text="Live RPi Camera (RGB)", padding="10")
        cam_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cam_label = tk.Label(cam_frame, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        self.btn_capture = ttk.Button(right_frame, text="CAPTURE RGB IMAGE", command=self.capture_image)
        self.btn_capture.pack(fill=tk.X, pady=5)
        
        # Video Recording Controls
        video_frame = ttk.Frame(right_frame)
        video_frame.pack(fill=tk.X, pady=5)
        
        self.btn_rec_start = ttk.Button(video_frame, text="START RECORD", command=self.start_recording)
        self.btn_rec_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.btn_rec_stop = ttk.Button(video_frame, text="STOP RECORD", command=self.stop_recording)
        self.btn_rec_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.btn_rec_stop.configure(state="disabled")

        self.btn_swap_color = ttk.Button(right_frame, text="TOGGLE COLOR FIX (SWAP R/B)", command=self.toggle_color)
        self.btn_swap_color.pack(fill=tk.X, pady=5)

        # White Balance toggle button
        self.btn_wb = ttk.Button(right_frame, text="TOGGLE WHITE BALANCE FIX (ON)", command=self.toggle_wb)
        self.btn_wb.pack(fill=tk.X, pady=5)

        # ----------------- KEYBOARD CONTROL -----------------
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.current_steer = 0.0
        self.target_steer = 0.0
        
        # Control Loop Constants
        self.SPEED_STEP = 30.0
        self.STEER_STEP = 3.0
        self.MAX_SPEED = 200.0
        self.MAX_STEER = 20.0
        
        # ----------------- CAMERA INITIALIZATION -----------------
        self.picam2 = None
        self.latest_frame = None

        # swap_rb: set True when using BGR888 format so Pillow gets proper RGB
        self.swap_rb = True

        # apply_wb: enables LAB-space white balance correction to remove bluish tint
        self.apply_wb = True
        
        # Video Recording State
        self.video_writer = None
        self.is_recording = False
        
        if _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                # Use BGR888 — more reliable on RPi 5 with newer libcamera.
                # swap_rb=True above will convert BGR -> RGB for display.
                cfg = self.picam2.create_video_configuration(
                    main={"size": (640, 480), "format": "BGR888"}
                )
                self.picam2.configure(cfg)
                self.picam2.start()
                logging.info("Picamera2 started successfully in BGR888 mode (will be converted to RGB).")
            except Exception as e:
                logging.error(f"Failed to start camera: {e}")
                self.picam2 = None

        self.start_control_loop()
        self.update_camera_feed()

    # --- WHITE BALANCE FIX ---
    def fix_white_balance(self, frame):
        """
        Correct bluish tint using LAB color space gray-world assumption.
        Works on RGB uint8 numpy arrays. Returns corrected RGB array.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        # Shift A and B channels toward neutral gray (128)
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # --- CAMERA METHODS ---
    def update_camera_feed(self):
        if self.picam2:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    # Convert BGR -> RGB if needed (required when using BGR888 format)
                    if self.swap_rb:
                        frame_rgb = frame[:, :, ::-1]
                    else:
                        frame_rgb = frame

                    # Apply white balance correction to remove bluish tint
                    if self.apply_wb:
                        frame_rgb = self.fix_white_balance(frame_rgb)

                    self.latest_frame = frame_rgb
                    
                    # Record frame if active
                    if self.is_recording and self.video_writer is not None:
                        # OpenCV VideoWriter expects BGR format (this is just the disk writing pipe)
                        bgr_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        self.video_writer.write(bgr_frame)
                        
                    img = Image.fromarray(frame_rgb, 'RGB')
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.cam_label.imgtk = imgtk
                    self.cam_label.configure(image=imgtk)
            except Exception as e:
                logging.debug(f"Frame drop: {e}")
        else:
            # Render a dummy frame if running offline
            mock = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock, "NO CAMERA (MOCK RUN)", (120, 240),
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
            logging.info(f"Image successfully saved: {filename}")
            messagebox.showinfo("Capture Saved", f"Image saved locally to:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame available to capture.")

    def start_recording(self):
        if not self.picam2:
            messagebox.showerror("Error", "Camera is not available.")
            return
            
        if self.is_recording:
            return
            
        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Using 640x480 resolution as configured in picamera, 20 fps matching UI loop roughly
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        
        if self.video_writer.isOpened():
            self.is_recording = True
            self.btn_rec_start.configure(state="disabled")
            self.btn_rec_stop.configure(state="normal")
            logging.info(f"Started recording video to: {filename}")
        else:
            messagebox.showerror("Error", "Failed to initialize video writer.")

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            
            self.btn_rec_start.configure(state="normal")
            self.btn_rec_stop.configure(state="disabled")
            logging.info("Video recording stopped and saved.")
            messagebox.showinfo("Recording Saved", "Video file has been saved successfully.")

    def toggle_color(self):
        self.swap_rb = not self.swap_rb
        logging.info(f"Swap R/B set to: {self.swap_rb}")

    def toggle_wb(self):
        self.apply_wb = not self.apply_wb
        state_str = "ON" if self.apply_wb else "OFF"
        self.btn_wb.config(text=f"TOGGLE WHITE BALANCE FIX ({state_str})")
        logging.info(f"White balance fix set to: {self.apply_wb}")

    # --- CONTROL LOOP ---
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

    # --- CONNECTION ---
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
        if self.is_recording:
            self.stop_recording()
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