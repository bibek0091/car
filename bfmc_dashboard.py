import sys
import cv2
import numpy as np
import time
from queue import Queue, Empty
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                             QGroupBox, QRadioButton, QSlider, QGridLayout, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

class BFMCDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFMC Professional Autonomous Dashboard")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        # Thread-safe data queues
        self.frame_queue = Queue(maxsize=2)
        self.radar_queue = Queue(maxsize=2)
        self.tlm_queue   = Queue(maxsize=5)
        
        self._init_ui()
        
        # Pull data from queues at ~30Hz
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_gui_from_queues)
        self.update_timer.start(33)
        
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        top_split = QHBoxLayout()
        
        # -----------------------------------------------------
        # LEFT: CAMERA FEED (60%)
        # -----------------------------------------------------
        self.cam_label = QLabel("INITIALIZING SENSORS...")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #000000; border: 2px solid #0078d4;")
        self.cam_label.setMinimumWidth(int(1280 * 0.6))
        top_split.addWidget(self.cam_label, stretch=6)
        
        # -----------------------------------------------------
        # RIGHT: TELEMETRY (40%)
        # -----------------------------------------------------
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #252526; border-radius: 5px;")
        right_layout = QVBoxLayout(right_panel)
        
        title = QLabel("TELEMETRY SENSORS")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #0078d4;")
        right_layout.addWidget(title)
        
        self.lbl_gps = QLabel("GPS: LAT 46.771200 | LON 23.623600")
        self.lbl_gps.setFont(QFont("Consolas", 12))
        self.lbl_gps.setStyleSheet("color: #aaaaaa;")
        right_layout.addWidget(self.lbl_gps)
        
        grid = QGridLayout()
        self.lbl_speed = QLabel("SPEED\n0.0 cm/s")
        self.lbl_speed.setFont(QFont("Consolas", 18, QFont.Bold))
        self.lbl_steer = QLabel("STEER\n0.0°")
        self.lbl_steer.setFont(QFont("Consolas", 18, QFont.Bold))
        grid.addWidget(self.lbl_speed, 0, 0)
        grid.addWidget(self.lbl_steer, 0, 1)
        right_layout.addLayout(grid)
        
        right_layout.addWidget(QLabel("BATTERY CAPACITY:"))
        self.prog_battery = QProgressBar()
        self.prog_battery.setValue(100)
        self.prog_battery.setStyleSheet("""
            QProgressBar { border: 1px solid #555; border-radius: 2px; text-align: center; }
            QProgressBar::chunk { background-color: #0078d4; }
        """)
        right_layout.addWidget(self.prog_battery)
        
        self.lbl_imu = QLabel("IMU: PITCH 0.0 | ROLL 0.0 | YAW 0.0")
        self.lbl_imu.setFont(QFont("Consolas", 11))
        self.lbl_imu.setStyleSheet("color: #aaaaaa;")
        right_layout.addWidget(self.lbl_imu)
        
        self.lbl_sys = QLabel("SYS: CPU 12% | RAM 40% | TEMP 45°C")
        self.lbl_sys.setFont(QFont("Consolas", 11))
        self.lbl_sys.setStyleSheet("color: #aaaaaa;")
        right_layout.addWidget(self.lbl_sys)
        
        right_layout.addSpacing(20)
        ds_title = QLabel("AI VISION ENGINE")
        ds_title.setFont(QFont("Arial", 14, QFont.Bold))
        ds_title.setStyleSheet("color: #0078d4;")
        right_layout.addWidget(ds_title)
        
        self.lbl_traffic = QLabel("TRAFFIC STATE: SYS_GO")
        self.lbl_traffic.setFont(QFont("Consolas", 14, QFont.Bold))
        self.lbl_traffic.setStyleSheet("color: #00ff00;")
        right_layout.addWidget(self.lbl_traffic)
        
        self.lbl_nav = QLabel("NAV MODE: RUNNING")
        self.lbl_nav.setFont(QFont("Consolas", 12))
        right_layout.addWidget(self.lbl_nav)
        
        self.lbl_light = QLabel("SIGNAL: [NONE]")
        self.lbl_light.setFont(QFont("Consolas", 12))
        right_layout.addWidget(self.lbl_light)
        
        led_layout = QHBoxLayout()
        self.led_stop = QLabel("STOP")
        self.led_stop.setAlignment(Qt.AlignCenter)
        self.led_stop.setStyleSheet("background-color: #440000; color: #888; padding: 5px; border-radius: 3px;")
        
        self.led_ped = QLabel("PEDESTRIAN")
        self.led_ped.setAlignment(Qt.AlignCenter)
        self.led_ped.setStyleSheet("background-color: #444400; color: #888; padding: 5px; border-radius: 3px;")
        
        self.led_lane = QLabel("LANE KEEP")
        self.led_lane.setAlignment(Qt.AlignCenter)
        self.led_lane.setStyleSheet("background-color: darkgreen; color: white; padding: 5px; border-radius: 3px;")
        
        led_layout.addWidget(self.led_stop)
        led_layout.addWidget(self.led_ped)
        led_layout.addWidget(self.led_lane)
        right_layout.addLayout(led_layout)
        
        right_layout.addStretch()
        top_split.addWidget(right_panel, stretch=4)
        main_layout.addLayout(top_split, stretch=8)
        
        # -----------------------------------------------------
        # BOTTOM: CONTROL PANEL (20%)
        # -----------------------------------------------------
        ctrl_panel = QFrame()
        ctrl_panel.setStyleSheet("background-color: #2d2d30; border-radius: 5px;")
        ctrl_layout = QHBoxLayout(ctrl_panel)
        
        self.btn_estop = QPushButton("EMERGENCY STOP (SPACE)")
        self.btn_estop.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_estop.setStyleSheet("background-color: #cc0000; color: white; padding: 15px; border-radius: 5px;")
        self.btn_estop.clicked.connect(lambda: print("ESTOP PRESSED"))
        
        self.btn_start = QPushButton("START AUTONOMY")
        self.btn_start.setStyleSheet("background-color: #0078d4; color: white; padding: 15px; border-radius: 5px;")
        
        ctrl_layout.addWidget(QLabel("Driving Mode:"))
        self.radio_man = QRadioButton("Manual")
        self.radio_auto = QRadioButton("Autonomous")
        self.radio_auto.setChecked(True)
        ctrl_layout.addWidget(self.radio_man)
        ctrl_layout.addWidget(self.radio_auto)
        
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel("Max Speed Target"))
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(0)
        speed_slider.setMaximum(100)
        speed_slider.setValue(80)
        slider_layout.addWidget(speed_slider)
        ctrl_layout.addLayout(slider_layout)
        
        ctrl_layout.addSpacing(40)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_estop)
        
        main_layout.addLayout(ctrl_panel, stretch=2)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            print("ESTOP TRIGGERED DIRECTLY VIA SPACEBAR!")
            self.lbl_traffic.setText("TRAFFIC STATE: SYS_STOP")
            self.lbl_traffic.setStyleSheet("color: red;")
        elif event.key() == Qt.Key_M:
            if self.radio_auto.isChecked(): self.radio_man.setChecked(True)
            else: self.radio_auto.setChecked(True)

    # -----------------------------------------------------------------
    # Safe methods for external threads to push data
    # -----------------------------------------------------------------
    def push_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put_nowait(frame.copy())
            
    def push_radar(self, frame):
        if not self.radar_queue.full():
            self.radar_queue.put_nowait(frame.copy())
            
    def push_telemetry(self, data):
        if not self.tlm_queue.full():
            self.tlm_queue.put_nowait(data)

    def _update_gui_from_queues(self):
        # 1. Update Camera Canvas
        try:
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            if frame is not None:
                # Composite radar if available
                radar = None
                try:
                    while not self.radar_queue.empty():
                        radar = self.radar_queue.get_nowait()
                except Empty: pass
                
                if radar is not None:
                    rh, rw = radar.shape[:2]
                    radar_resized = cv2.resize(radar, (int(rw*0.5), int(rh*0.5)))
                    rrh, rrw = radar_resized.shape[:2]
                    # Draw bright white border around radar
                    cv2.rectangle(radar_resized, (0,0), (rrw-1, rrh-1), (255,255,255), 3)
                    frame[-rrh-20:-20, 20:rrw+20] = radar_resized 
                    cv2.putText(frame, "RADAR", (30, frame.shape[0]-rrh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                h, w, ch = frame.shape
                bytes_per_line = ch * w
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled = QPixmap.fromImage(qt_img).scaled(self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio)
                self.cam_label.setPixmap(scaled)
        except Empty: pass
            
        # 2. Update Telemetry values
        try:
            data = None
            while not self.tlm_queue.empty():
                data = self.tlm_queue.get_nowait()
            if data:
                self.lbl_speed.setText(f"SPEED\n{abs(data.get('speed', 0)):.1f} cm/s")
                self.lbl_steer.setText(f"STEER\n{data.get('steering', 0):+.1f}°")
                self.prog_battery.setValue(int(data.get('battery', 100)))
                self.lbl_gps.setText(f"GPS: LAT {data.get('lat', 46.7712):.6f} | LON {data.get('lon', 23.6236):.6f}")
                self.lbl_imu.setText(f"IMU: PITCH {data.get('pitch', 0):.1f} | ROLL {data.get('roll', 0):.1f} | YAW {data.get('yaw', 0):.1f}")
                
                t_state = data.get('traffic', 'SYS_GO')
                if t_state == "SYS_STOP": 
                    self.lbl_traffic.setText(f"TRAFFIC: STOP [{data.get('reason','')}]")
                    self.lbl_traffic.setStyleSheet("color: #ff3333;")
                elif t_state == "SYS_SLOW": 
                    self.lbl_traffic.setText(f"TRAFFIC: SLOW [{data.get('reason','')}]")
                    self.lbl_traffic.setStyleSheet("color: #ffaa00;")
                else: 
                    self.lbl_traffic.setText("TRAFFIC: GO (CLEAR)")
                    self.lbl_traffic.setStyleSheet("color: #00ff00;")
                
                self.lbl_nav.setText(f"NAV MODE: {data.get('nav', 'NORMAL')} | Anchor: {data.get('anchor', 'N/A')}")
                
                ls = data.get('light', '[NONE]')
                self.lbl_light.setText(f"SIGNAL: {ls}")
                if "RED" in ls: self.lbl_light.setStyleSheet("color: #ff3333;")
                elif "GREEN" in ls: self.lbl_light.setStyleSheet("color: #00ff00;")
                else: self.lbl_light.setStyleSheet("color: white;")
                
                reason = data.get('reason', '')
                if "STOP SIGN" in reason:
                    self.led_stop.setStyleSheet("background-color: #ff0000; color: white; padding: 5px; border-radius: 3px;")
                else:
                    self.led_stop.setStyleSheet("background-color: #440000; color: #888; padding: 5px; border-radius: 3px;")
                    
                if "pedestrian" in reason.lower():
                    self.led_ped.setStyleSheet("background-color: #ffff00; color: black; padding: 5px; border-radius: 3px;")
                else:
                    self.led_ped.setStyleSheet("background-color: #444400; color: #888; padding: 5px; border-radius: 3px;")
                    
                if data.get('anchor', '') != "LOST":
                    self.led_lane.setStyleSheet("background-color: darkgreen; color: white; padding: 5px; border-radius: 3px;")
                else:
                    self.led_lane.setStyleSheet("background-color: #004400; color: #888; padding: 5px; border-radius: 3px;")
        except Empty: pass
