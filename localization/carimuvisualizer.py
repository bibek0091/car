"""
Car IMU Visualizer — MPU-9250 + Raspberry Pi
=============================================
Visualizes a 3-panel car model responding to real IMU data:
  - Top view    : shows YAW   (turning left/right)
  - Front view  : shows ROLL  (tilting left/right)
  - Side view   : shows PITCH (tilting forward/back)

Install:
    pip install mpu9250-jmdev numpy pygame

Run:
    python car_imu_visualizer.py
"""

import numpy as np
import pygame
import time
import math
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250

# ─────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────
BLACK      = (10,  10,  10)
WHITE      = (240, 240, 240)
BG         = (20,  20,  30)
PANEL_BG   = (30,  30,  45)
CAR_BODY   = (52,  152, 219)   # blue
CAR_ROOF   = (41,  128, 185)
WHEEL      = (44,  44,  44)
WHEEL_RIM  = (180, 180, 180)
WINDSHIELD = (163, 228, 215)
ARROW      = (46,  204, 113)   # green
RED        = (231, 76,  60)
YELLOW     = (241, 196, 15)
GRAY       = (100, 100, 120)
GRID       = (40,  40,  60)
TEXT_COLOR = (220, 220, 220)
ACCENT     = (155, 89,  182)

# ─────────────────────────────────────────────
#  MADGWICK FILTER
# ─────────────────────────────────────────────
class MadgwickFilter:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
        q = self.q
        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if norm == 0: return
        ax, ay, az = ax/norm, ay/norm, az/norm

        norm = math.sqrt(mx**2 + my**2 + mz**2)
        if norm == 0: return
        mx, my, mz = mx/norm, my/norm, mz/norm

        h = np.array([
            2*mx*(0.5-q[2]**2-q[3]**2)+2*my*(q[1]*q[2]-q[0]*q[3])+2*mz*(q[1]*q[3]+q[0]*q[2]),
            2*mx*(q[1]*q[2]+q[0]*q[3])+2*my*(0.5-q[1]**2-q[3]**2)+2*mz*(q[2]*q[3]-q[0]*q[1]),
            2*mx*(q[1]*q[3]-q[0]*q[2])+2*my*(q[2]*q[3]+q[0]*q[1])+2*mz*(0.5-q[1]**2-q[2]**2)
        ])
        bx = math.sqrt(h[0]**2 + h[1]**2)
        bz = h[2]

        F = np.array([
            2*(q[1]*q[3]-q[0]*q[2])-ax,
            2*(q[0]*q[1]+q[2]*q[3])-ay,
            2*(0.5-q[1]**2-q[2]**2)-az,
            2*bx*(0.5-q[2]**2-q[3]**2)+2*bz*(q[1]*q[3]-q[0]*q[2])-mx,
            2*bx*(q[1]*q[2]-q[0]*q[3])+2*bz*(q[0]*q[1]+q[2]*q[3])-my,
            2*bx*(q[0]*q[2]+q[1]*q[3])+2*bz*(0.5-q[1]**2-q[2]**2)-mz
        ])
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [0,      -4*q[1], -4*q[2], 0],
            [-2*bz*q[2], 2*bz*q[3], -4*bx*q[2]-2*bz*q[0], -4*bx*q[3]+2*bz*q[1]],
            [-2*bx*q[3]+2*bz*q[1], 2*bx*q[2]+2*bz*q[0], 2*bx*q[1]+2*bz*q[3], -2*bx*q[0]+2*bz*q[2]],
            [2*bx*q[2], 2*bx*q[3]-4*bz*q[1], 2*bx*q[0]-4*bz*q[2], 2*bx*q[1]]
        ])

        step = J.T @ F
        norm = np.linalg.norm(step)
        if norm == 0: return
        step /= norm

        qDot = 0.5 * np.array([
            -q[1]*gx - q[2]*gy - q[3]*gz,
             q[0]*gx + q[2]*gz - q[3]*gy,
             q[0]*gy - q[1]*gz + q[3]*gx,
             q[0]*gz + q[1]*gy - q[2]*gx
        ]) - self.beta * step

        self.q += qDot * dt
        self.q /= np.linalg.norm(self.q)

    def get_euler(self):
        q = self.q
        roll  = math.degrees(math.atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2*(q[0]*q[2]-q[3]*q[1])))))
        yaw   = math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
        return roll, pitch, yaw


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def rotate_point(px, py, cx, cy, angle_deg):
    """Rotate point (px,py) around center (cx,cy) by angle_deg."""
    r = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    nx = dx * math.cos(r) - dy * math.sin(r)
    ny = dx * math.sin(r) + dy * math.cos(r)
    return cx + nx, cy + ny

def rotate_polygon(points, cx, cy, angle_deg):
    return [rotate_point(p[0], p[1], cx, cy, angle_deg) for p in points]

def draw_grid(surface, rect, spacing=30):
    x0, y0, w, h = rect
    for x in range(x0, x0+w, spacing):
        pygame.draw.line(surface, GRID, (x, y0), (x, y0+h), 1)
    for y in range(y0, y0+h, spacing):
        pygame.draw.line(surface, GRID, (x0, y), (x0+w, y), 1)

def draw_panel(surface, rect, title, font):
    x, y, w, h = rect
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=12)
    pygame.draw.rect(surface, GRAY, rect, 2, border_radius=12)
    draw_grid(surface, rect)
    label = font.render(title, True, ACCENT)
    surface.blit(label, (x + 10, y + 8))

def draw_text_center(surface, text, font, color, cx, cy):
    t = font.render(text, True, color)
    r = t.get_rect(center=(cx, cy))
    surface.blit(t, r)

def draw_angle_arc(surface, cx, cy, radius, angle, color, label, font_small):
    """Draw an arc and label showing the angle."""
    pygame.draw.arc(surface, color,
                    (cx-radius, cy-radius, radius*2, radius*2),
                    math.radians(-90), math.radians(-90 + angle) if angle >= 0 else math.radians(-90 + angle),
                    3)
    # Angle label
    t = font_small.render(f"{angle:.1f}°", True, color)
    surface.blit(t, (cx + radius + 5, cy - 10))


# ─────────────────────────────────────────────
#  CAR DRAWING FUNCTIONS
# ─────────────────────────────────────────────

def draw_car_top(surface, cx, cy, yaw, font_small):
    """Top-down view — shows YAW (turning)."""
    W, H = 36, 70  # car width, height
    roof_w, roof_h = 28, 38

    # Car body
    body = [
        (cx - W//2, cy - H//2),
        (cx + W//2, cy - H//2),
        (cx + W//2 + 6, cy - H//4),
        (cx + W//2 + 6, cy + H//4),
        (cx + W//2, cy + H//2),
        (cx - W//2, cy + H//2),
        (cx - W//2 - 6, cy + H//4),
        (cx - W//2 - 6, cy - H//4),
    ]
    body = rotate_polygon(body, cx, cy, yaw)
    pygame.draw.polygon(surface, CAR_BODY, body)
    pygame.draw.polygon(surface, WHITE, body, 2)

    # Roof
    roof = [
        (cx - roof_w//2, cy - roof_h//2),
        (cx + roof_w//2, cy - roof_h//2),
        (cx + roof_w//2, cy + roof_h//2),
        (cx - roof_w//2, cy + roof_h//2),
    ]
    roof = rotate_polygon(roof, cx, cy, yaw)
    pygame.draw.polygon(surface, CAR_ROOF, roof)

    # Windshield (front)
    ws = [
        (cx - 12, cy - H//2 + 5),
        (cx + 12, cy - H//2 + 5),
        (cx + 10, cy - H//2 + 16),
        (cx - 10, cy - H//2 + 16),
    ]
    ws = rotate_polygon(ws, cx, cy, yaw)
    pygame.draw.polygon(surface, WINDSHIELD, ws)

    # Front arrow indicator
    arrow_tip = rotate_point(cx, cy - H//2 - 18, cx, cy, yaw)
    arrow_l   = rotate_point(cx - 8, cy - H//2 - 6, cx, cy, yaw)
    arrow_r   = rotate_point(cx + 8, cy - H//2 - 6, cx, cy, yaw)
    pygame.draw.polygon(surface, ARROW, [arrow_tip, arrow_l, arrow_r])

    # Wheels
    wheel_positions = [
        (cx - W//2 - 2, cy - H//2 + 10),  # front-left
        (cx + W//2 + 2, cy - H//2 + 10),  # front-right
        (cx - W//2 - 2, cy + H//2 - 10),  # rear-left
        (cx + W//2 + 2, cy + H//2 - 10),  # rear-right
    ]
    for wx, wy in wheel_positions:
        rx, ry = rotate_point(wx, wy, cx, cy, yaw)
        pygame.draw.ellipse(surface, WHEEL, (rx-5, ry-8, 10, 16))
        pygame.draw.ellipse(surface, WHEEL_RIM, (rx-3, ry-5, 6, 10), 1)

    # Heading label
    t = font_small.render(f"YAW: {yaw:.1f}°", True, YELLOW)
    surface.blit(t, (cx - 30, cy + H//2 + 20))


def draw_car_front(surface, cx, cy, roll, font_small):
    """Front view — shows ROLL (tilting left/right)."""
    W, H = 80, 50   # car width, height in front view
    roof_w = 50

    # Car body
    body = [
        (cx - W//2,      cy + H//2),
        (cx + W//2,      cy + H//2),
        (cx + W//2 + 10, cy),
        (cx + W//2 - 5,  cy - H//2 + 10),
        (cx - W//2 + 5,  cy - H//2 + 10),
        (cx - W//2 - 10, cy),
    ]
    body = rotate_polygon(body, cx, cy, roll)
    pygame.draw.polygon(surface, CAR_BODY, body)
    pygame.draw.polygon(surface, WHITE, body, 2)

    # Roof
    roof = [
        (cx - roof_w//2, cy - H//2 + 10),
        (cx + roof_w//2, cy - H//2 + 10),
        (cx + roof_w//2 - 5, cy - H//2 - 18),
        (cx - roof_w//2 + 5, cy - H//2 - 18),
    ]
    roof = rotate_polygon(roof, cx, cy, roll)
    pygame.draw.polygon(surface, CAR_ROOF, roof)

    # Windshield
    ws = [
        (cx - 20, cy - H//2 + 10),
        (cx + 20, cy - H//2 + 10),
        (cx + 16, cy - H//2 - 16),
        (cx - 16, cy - H//2 - 16),
    ]
    ws = rotate_polygon(ws, cx, cy, roll)
    pygame.draw.polygon(surface, WINDSHIELD, ws)

    # Wheels
    for wx in [cx - W//2 - 5, cx + W//2 + 5]:
        rx, ry = rotate_point(wx, cy + H//2, cx, cy, roll)
        pygame.draw.ellipse(surface, WHEEL, (rx-12, ry-8, 24, 16))
        pygame.draw.circle(surface, WHEEL_RIM, (int(rx), int(ry)), 5, 2)

    # Roll horizon line
    lx1, ly1 = rotate_point(cx - 60, cy + H//2 + 15, cx, cy + H//2 + 15, roll)
    lx2, ly2 = rotate_point(cx + 60, cy + H//2 + 15, cx, cy + H//2 + 15, roll)
    pygame.draw.line(surface, RED, (int(lx1), int(ly1)), (int(lx2), int(ly2)), 2)

    t = font_small.render(f"ROLL: {roll:.1f}°", True, YELLOW)
    surface.blit(t, (cx - 35, cy + H//2 + 30))


def draw_car_side(surface, cx, cy, pitch, font_small):
    """Side view — shows PITCH (tilting forward/back)."""
    W, H = 100, 45

    # Car body
    body = [
        (cx - W//2,      cy + H//2),
        (cx + W//2,      cy + H//2),
        (cx + W//2 + 5,  cy + H//4),
        (cx + W//2,      cy - H//2 + 10),
        (cx + W//4,      cy - H//2 - 5),
        (cx - W//4 + 5,  cy - H//2 - 5),
        (cx - W//2 + 5,  cy - H//2 + 10),
        (cx - W//2 - 5,  cy + H//4),
    ]
    body = rotate_polygon(body, cx, cy, pitch)
    pygame.draw.polygon(surface, CAR_BODY, body)
    pygame.draw.polygon(surface, WHITE, body, 2)

    # Roof
    roof = [
        (cx - W//4 + 5,  cy - H//2 - 5),
        (cx + W//4,      cy - H//2 - 5),
        (cx + W//4 - 5,  cy - H//2 - 22),
        (cx - W//4 + 8,  cy - H//2 - 22),
    ]
    roof = rotate_polygon(roof, cx, cy, pitch)
    pygame.draw.polygon(surface, CAR_ROOF, roof)

    # Windshield
    ws = [
        (cx - W//4 + 8,  cy - H//2 - 22),
        (cx + W//4 - 5,  cy - H//2 - 22),
        (cx + W//4,      cy - H//2 - 5),
        (cx - W//4 + 5,  cy - H//2 - 5),
    ]
    ws = rotate_polygon(ws, cx, cy, pitch)
    pygame.draw.polygon(surface, WINDSHIELD, ws)

    # Wheels
    for wx in [cx - W//2 + 12, cx + W//2 - 12]:
        rx, ry = rotate_point(wx, cy + H//2 + 2, cx, cy, pitch)
        pygame.draw.circle(surface, WHEEL, (int(rx), int(ry)), 13)
        pygame.draw.circle(surface, WHEEL_RIM, (int(rx), int(ry)), 7, 2)

    # Ground line
    pygame.draw.line(surface, GRAY,
                     (cx - W//2 - 20, cy + H//2 + 15),
                     (cx + W//2 + 20, cy + H//2 + 15), 2)

    t = font_small.render(f"PITCH: {pitch:.1f}°", True, YELLOW)
    surface.blit(t, (cx - 38, cy + H//2 + 25))


# ─────────────────────────────────────────────
#  MPU-9250 SETUP
# ─────────────────────────────────────────────
def setup_mpu():
    mpu = MPU9250(
        address_ak=AK8963_ADDRESS,
        address_mpu_master=MPU9050_ADDRESS_68,
        bus=1,
        gfs=GFS_1000,
        afs=AFS_8G,
        mfs=AK8963_BIT_16,
        mode=AK8963_MODE_C100HZ
    )
    mpu.configure()
    return mpu


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # Init IMU
    print("Connecting to MPU-9250...")
    mpu = setup_mpu()
    madgwick = MadgwickFilter(beta=0.1)
    print("MPU-9250 ready!")

    # Calibrate
    print("Calibrating (keep still for 2 seconds)...")
    for _ in range(200):
        accel = mpu.readAccelerometerMaster()
        gyro  = mpu.readGyroscopeMaster()
        mag   = mpu.readMagnetometerMaster()
        if None in accel or None in gyro or None in mag: continue
        ax, ay, az = accel
        gx, gy, gz = [math.radians(v) for v in gyro]
        mx, my, mz = mag
        madgwick.update(ax, ay, az, gx, gy, gz, mx, my, mz, dt=0.01)
        time.sleep(0.01)
    print("Calibration done!")

    # Init pygame
    pygame.init()
    W_SCREEN, H_SCREEN = 1100, 700
    screen = pygame.display.set_mode((W_SCREEN, H_SCREEN))
    pygame.display.set_caption("Car IMU Visualizer — Bosch Challenge")
    clock = pygame.time.Clock()

    font_title  = pygame.font.SysFont("monospace", 22, bold=True)
    font_label  = pygame.font.SysFont("monospace", 16, bold=True)
    font_small  = pygame.font.SysFont("monospace", 14)
    font_data   = pygame.font.SysFont("monospace", 15)

    roll = pitch = yaw = 0.0
    prev_time = time.time()

    # Yaw history for mini trail
    yaw_history = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                madgwick.q = np.array([1.0, 0.0, 0.0, 0.0])

        # Read IMU
        now = time.time()
        dt  = now - prev_time
        prev_time = now

        accel = mpu.readAccelerometerMaster()
        gyro  = mpu.readGyroscopeMaster()
        mag   = mpu.readMagnetometerMaster()

        if None not in accel and None not in gyro and None not in mag:
            ax, ay, az = accel
            gx, gy, gz = [math.radians(v) for v in gyro]
            mx, my, mz = mag
            madgwick.update(ax, ay, az, gx, gy, gz, mx, my, mz, dt)
            roll, pitch, yaw = madgwick.get_euler()

        yaw_history.append(yaw)
        if len(yaw_history) > 100:
            yaw_history.pop(0)

        # ── Draw ──
        screen.fill(BG)

        # Title bar
        pygame.draw.rect(screen, (25, 25, 40), (0, 0, W_SCREEN, 50))
        title = font_title.render("🚗  Car IMU Visualizer  |  MPU-9250 + Madgwick  |  Bosch Future Mobility", True, WHITE)
        screen.blit(title, (20, 14))
        hint = font_small.render("Press R to reset orientation", True, GRAY)
        screen.blit(hint, (W_SCREEN - 230, 17))

        # ── Panel 1: TOP VIEW (YAW) ──
        p1 = (20, 60, 320, 360)
        draw_panel(screen, p1, "TOP VIEW — YAW (Turning)", font_label)
        draw_car_top(screen, 180, 250, yaw, font_small)

        # Compass rose
        comp_cx, comp_cy = 280, 110
        pygame.draw.circle(screen, GRAY, (comp_cx, comp_cy), 28, 1)
        for label, angle in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            lx = comp_cx + 35 * math.sin(math.radians(angle))
            ly = comp_cy - 35 * math.cos(math.radians(angle))
            draw_text_center(screen, label, font_small, GRAY, int(lx), int(ly))
        nx = comp_cx + 22 * math.sin(math.radians(yaw))
        ny = comp_cy - 22 * math.cos(math.radians(yaw))
        pygame.draw.line(screen, RED, (comp_cx, comp_cy), (int(nx), int(ny)), 3)
        pygame.draw.circle(screen, RED, (comp_cx, comp_cy), 4)

        # ── Panel 2: FRONT VIEW (ROLL) ──
        p2 = (360, 60, 320, 360)
        draw_panel(screen, p2, "FRONT VIEW — ROLL (Tilt L/R)", font_label)
        draw_car_front(screen, 520, 240, roll, font_small)

        # Roll bar
        bar_cx, bar_cy = 490, 380
        pygame.draw.line(screen, GRAY, (bar_cx - 80, bar_cy), (bar_cx + 80, bar_cy), 2)
        pygame.draw.circle(screen, GRAY, (bar_cx, bar_cy), 5)
        roll_x = bar_cx + int(min(max(roll, -45), 45) / 45 * 75)
        pygame.draw.line(screen, RED, (bar_cx, bar_cy), (roll_x, bar_cy - 15), 3)
        pygame.draw.circle(screen, RED, (roll_x, bar_cy - 15), 5)

        # ── Panel 3: SIDE VIEW (PITCH) ──
        p3 = (700, 60, 380, 360)
        draw_panel(screen, p3, "SIDE VIEW — PITCH (Tilt F/B)", font_label)
        draw_car_side(screen, 890, 240, -pitch, font_small)

        # ── Data Panel ──
        dp = (20, 440, 1060, 240)
        draw_panel(screen, dp, "LIVE IMU DATA", font_label)

        # Gauge bars
        def draw_gauge(label, value, min_v, max_v, x, y, color):
            w = 200
            pct = (value - min_v) / (max_v - min_v)
            pct = max(0, min(1, pct))
            pygame.draw.rect(screen, GRAY, (x, y, w, 18), border_radius=4)
            pygame.draw.rect(screen, color, (x, y, int(w * pct), 18), border_radius=4)
            pygame.draw.rect(screen, WHITE, (x, y, w, 18), 1, border_radius=4)
            lbl = font_data.render(f"{label}: {value:+.1f}°", True, WHITE)
            screen.blit(lbl, (x + w + 10, y))

        draw_gauge("ROLL ",  roll,  -90,  90, 40,  490, CAR_BODY)
        draw_gauge("PITCH", pitch, -90,  90, 40,  525, ARROW)
        draw_gauge("YAW  ",  yaw,  -180, 180, 40, 560, YELLOW)

        # Yaw history graph
        gx0, gy0, gw, gh = 320, 455, 400, 120
        pygame.draw.rect(screen, (20, 20, 35), (gx0, gy0, gw, gh))
        pygame.draw.rect(screen, GRAY, (gx0, gy0, gw, gh), 1)
        mid_y = gy0 + gh // 2
        pygame.draw.line(screen, GRID, (gx0, mid_y), (gx0+gw, mid_y), 1)
        t = font_small.render("Yaw History", True, GRAY)
        screen.blit(t, (gx0 + 5, gy0 + 3))
        if len(yaw_history) > 1:
            pts = []
            for i, y_val in enumerate(yaw_history):
                px = gx0 + int(i / 100 * gw)
                py = mid_y - int(y_val / 180 * (gh//2 - 5))
                pts.append((px, py))
            pygame.draw.lines(screen, YELLOW, False, pts, 2)

        # Numeric readout
        nx0 = 760
        screen.blit(font_label.render("ORIENTATION", True, ACCENT), (nx0, 460))
        for i, (lbl, val, col) in enumerate([
            ("Roll  ", roll,  CAR_BODY),
            ("Pitch ", pitch, ARROW),
            ("Yaw   ", yaw,   YELLOW),
        ]):
            txt = font_data.render(f"{lbl}: {val:+8.2f}°", True, col)
            screen.blit(txt, (nx0, 490 + i*28))

        # Status
        status = font_small.render("● LIVE", True, ARROW)
        screen.blit(status, (W_SCREEN - 80, H_SCREEN - 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Exited.")


if __name__ == "__main__":
    main()