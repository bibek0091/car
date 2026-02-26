"""
Car IMU Visualizer — MPU-9250 + Raspberry Pi
=============================================
Uses smbus2 directly — no AK8963 magnetometer dependency.
Complementary filter for Roll & Pitch. Gyro integration for Yaw.

Install:
    pip install smbus2 numpy pygame

Run:
    python car_imu_visualizer.py
"""

import numpy as np
import pygame
import time
import math
import smbus2

# ─────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────
WHITE      = (240, 240, 240)
BG         = (20,  20,  30)
PANEL_BG   = (30,  30,  45)
CAR_BODY   = (52,  152, 219)
CAR_ROOF   = (41,  128, 185)
WHEEL      = (44,  44,  44)
WHEEL_RIM  = (180, 180, 180)
WINDSHIELD = (163, 228, 215)
ARROW      = (46,  204, 113)
RED        = (231, 76,  60)
YELLOW     = (241, 196, 15)
GRAY       = (100, 100, 120)
GRID       = (40,  40,  60)
ACCENT     = (155, 89,  182)

# ─────────────────────────────────────────────
#  MPU-9250 DIRECT READER (smbus2)
# ─────────────────────────────────────────────
class MPU9250:
    def __init__(self, bus=1, address=0x68):
        self.bus  = smbus2.SMBus(bus)
        self.addr = address
        self.bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake up
        time.sleep(0.1)
        self.bus.write_byte_data(self.addr, 0x1B, 0x10)  # gyro ±1000 deg/s
        self.bus.write_byte_data(self.addr, 0x1C, 0x10)  # accel ±8g
        time.sleep(0.1)
        print(f"MPU-9250 connected at 0x{self.addr:02X}")

    def _signed(self, val):
        return val - 65536 if val > 32767 else val

    def read(self):
        d = self.bus.read_i2c_block_data(self.addr, 0x3B, 14)
        ax = self._signed(d[0]  << 8 | d[1])  / 4096.0   # ±8g
        ay = self._signed(d[2]  << 8 | d[3])  / 4096.0
        az = self._signed(d[4]  << 8 | d[5])  / 4096.0
        gx = self._signed(d[8]  << 8 | d[9])  / 32.8     # ±1000 deg/s
        gy = self._signed(d[10] << 8 | d[11]) / 32.8
        gz = self._signed(d[12] << 8 | d[13]) / 32.8
        return ax, ay, az, gx, gy, gz


# ─────────────────────────────────────────────
#  COMPLEMENTARY FILTER
# ─────────────────────────────────────────────
class ComplementaryFilter:
    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.roll  = 0.0
        self.pitch = 0.0
        self.yaw   = 0.0

    def update(self, ax, ay, az, gx, gy, gz, dt):
        roll_acc  = math.degrees(math.atan2(ay, az))
        pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        self.roll  = self.alpha * (self.roll  + gx * dt) + (1 - self.alpha) * roll_acc
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * pitch_acc
        self.yaw  += gz * dt
        if self.yaw >  180: self.yaw -= 360
        if self.yaw < -180: self.yaw += 360
        return self.roll, self.pitch, self.yaw

    def reset(self):
        self.roll = self.pitch = self.yaw = 0.0


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def rot(px, py, cx, cy, a):
    r  = math.radians(a)
    dx, dy = px-cx, py-cy
    return cx + dx*math.cos(r) - dy*math.sin(r), cy + dx*math.sin(r) + dy*math.cos(r)

def rot_poly(pts, cx, cy, a):
    return [rot(p[0], p[1], cx, cy, a) for p in pts]

def draw_grid(surf, rect):
    x0, y0, w, h = rect
    for x in range(x0, x0+w, 30):
        pygame.draw.line(surf, GRID, (x,y0), (x,y0+h))
    for y in range(y0, y0+h, 30):
        pygame.draw.line(surf, GRID, (x0,y), (x0+w,y))

def panel(surf, rect, title, font):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=12)
    pygame.draw.rect(surf, GRAY,     rect, 2, border_radius=12)
    draw_grid(surf, rect)
    surf.blit(font.render(title, True, ACCENT), (rect[0]+10, rect[1]+8))

def txt_c(surf, text, font, color, cx, cy):
    t = font.render(text, True, color)
    surf.blit(t, t.get_rect(center=(cx,cy)))


# ─────────────────────────────────────────────
#  CAR VIEWS
# ─────────────────────────────────────────────
def car_top(surf, cx, cy, yaw, fs):
    W, H = 36, 70
    body = rot_poly([(cx-W//2,cy-H//2),(cx+W//2,cy-H//2),
                     (cx+W//2+6,cy-H//4),(cx+W//2+6,cy+H//4),
                     (cx+W//2,cy+H//2),(cx-W//2,cy+H//2),
                     (cx-W//2-6,cy+H//4),(cx-W//2-6,cy-H//4)], cx, cy, yaw)
    pygame.draw.polygon(surf, CAR_BODY, body)
    pygame.draw.polygon(surf, WHITE, body, 2)
    roof = rot_poly([(cx-14,cy-19),(cx+14,cy-19),(cx+14,cy+19),(cx-14,cy+19)],cx,cy,yaw)
    pygame.draw.polygon(surf, CAR_ROOF, roof)
    ws = rot_poly([(cx-12,cy-H//2+5),(cx+12,cy-H//2+5),
                   (cx+10,cy-H//2+16),(cx-10,cy-H//2+16)],cx,cy,yaw)
    pygame.draw.polygon(surf, WINDSHIELD, ws)
    tip = rot(cx, cy-H//2-18, cx, cy, yaw)
    al  = rot(cx-8, cy-H//2-6, cx, cy, yaw)
    ar  = rot(cx+8, cy-H//2-6, cx, cy, yaw)
    pygame.draw.polygon(surf, ARROW, [tip, al, ar])
    for wx,wy in [(cx-W//2-2,cy-H//2+10),(cx+W//2+2,cy-H//2+10),
                  (cx-W//2-2,cy+H//2-10),(cx+W//2+2,cy+H//2-10)]:
        rx,ry = rot(wx,wy,cx,cy,yaw)
        pygame.draw.ellipse(surf, WHEEL,     (rx-5,ry-8,10,16))
        pygame.draw.ellipse(surf, WHEEL_RIM, (rx-3,ry-5,6,10),1)
    surf.blit(fs.render(f"YAW: {yaw:+.1f} deg", True, YELLOW),(cx-45,cy+H//2+18))

def car_front(surf, cx, cy, roll, fs):
    W, H = 80, 50
    body = rot_poly([(cx-W//2,cy+H//2),(cx+W//2,cy+H//2),
                     (cx+W//2+10,cy),(cx+W//2-5,cy-H//2+10),
                     (cx-W//2+5,cy-H//2+10),(cx-W//2-10,cy)],cx,cy,roll)
    pygame.draw.polygon(surf, CAR_BODY, body)
    pygame.draw.polygon(surf, WHITE, body, 2)
    roof = rot_poly([(cx-25,cy-H//2+10),(cx+25,cy-H//2+10),
                     (cx+20,cy-H//2-18),(cx-20,cy-H//2-18)],cx,cy,roll)
    pygame.draw.polygon(surf, CAR_ROOF, roof)
    ws = rot_poly([(cx-20,cy-H//2+10),(cx+20,cy-H//2+10),
                   (cx+16,cy-H//2-16),(cx-16,cy-H//2-16)],cx,cy,roll)
    pygame.draw.polygon(surf, WINDSHIELD, ws)
    for wx in [cx-W//2-5, cx+W//2+5]:
        rx,ry = rot(wx,cy+H//2,cx,cy,roll)
        pygame.draw.ellipse(surf, WHEEL,    (rx-12,ry-8,24,16))
        pygame.draw.circle(surf, WHEEL_RIM, (int(rx),int(ry)),5,2)
    l1 = rot(cx-60, cy+H//2+15, cx, cy+H//2+15, roll)
    l2 = rot(cx+60, cy+H//2+15, cx, cy+H//2+15, roll)
    pygame.draw.line(surf, RED, (int(l1[0]),int(l1[1])), (int(l2[0]),int(l2[1])), 2)
    surf.blit(fs.render(f"ROLL: {roll:+.1f} deg", True, YELLOW),(cx-50,cy+H//2+28))

def car_side(surf, cx, cy, pitch, fs):
    W, H = 100, 45
    body = rot_poly([(cx-W//2,cy+H//2),(cx+W//2,cy+H//2),
                     (cx+W//2+5,cy+H//4),(cx+W//2,cy-H//2+10),
                     (cx+W//4,cy-H//2-5),(cx-W//4+5,cy-H//2-5),
                     (cx-W//2+5,cy-H//2+10),(cx-W//2-5,cy+H//4)],cx,cy,pitch)
    pygame.draw.polygon(surf, CAR_BODY, body)
    pygame.draw.polygon(surf, WHITE, body, 2)
    roof = rot_poly([(cx-W//4+5,cy-H//2-5),(cx+W//4,cy-H//2-5),
                     (cx+W//4-5,cy-H//2-22),(cx-W//4+8,cy-H//2-22)],cx,cy,pitch)
    pygame.draw.polygon(surf, CAR_ROOF, roof)
    ws = rot_poly([(cx-W//4+8,cy-H//2-22),(cx+W//4-5,cy-H//2-22),
                   (cx+W//4,cy-H//2-5),(cx-W//4+5,cy-H//2-5)],cx,cy,pitch)
    pygame.draw.polygon(surf, WINDSHIELD, ws)
    for wx in [cx-W//2+12, cx+W//2-12]:
        rx,ry = rot(wx,cy+H//2+2,cx,cy,pitch)
        pygame.draw.circle(surf, WHEEL,     (int(rx),int(ry)),13)
        pygame.draw.circle(surf, WHEEL_RIM, (int(rx),int(ry)),7,2)
    pygame.draw.line(surf, GRAY,(cx-W//2-20,cy+H//2+15),(cx+W//2+20,cy+H//2+15),2)
    surf.blit(fs.render(f"PITCH: {pitch:+.1f} deg", True, YELLOW),(cx-55,cy+H//2+23))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("Connecting to MPU-9250...")
    mpu = MPU9250(bus=1, address=0x68)
    cf  = ComplementaryFilter(alpha=0.96)

    print("Calibrating gyro (keep still 2 sec)...")
    gx_b = gy_b = gz_b = 0.0
    N = 200
    for _ in range(N):
        _, _, _, gx, gy, gz = mpu.read()
        gx_b += gx; gy_b += gy; gz_b += gz
        time.sleep(0.01)
    gx_b /= N; gy_b /= N; gz_b /= N
    print(f"Bias  gx={gx_b:.3f}  gy={gy_b:.3f}  gz={gz_b:.3f}")

    pygame.init()
    SW, SH = 1100, 700
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Car IMU Visualizer — Bosch Challenge")
    clock = pygame.time.Clock()

    ft = pygame.font.SysFont("monospace", 20, bold=True)
    fl = pygame.font.SysFont("monospace", 15, bold=True)
    fs = pygame.font.SysFont("monospace", 13)
    fd = pygame.font.SysFont("monospace", 15)

    roll = pitch = yaw = 0.0
    yaw_hist = []
    prev = time.time()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                cf.reset()

        now = time.time()
        dt  = max(now - prev, 0.001)
        prev = now

        try:
            ax, ay, az, gx, gy, gz = mpu.read()
            gx -= gx_b; gy -= gy_b; gz -= gz_b
            roll, pitch, yaw = cf.update(ax, ay, az, gx, gy, gz, dt)
        except Exception as ex:
            print(f"Read error: {ex}")

        yaw_hist.append(yaw)
        if len(yaw_hist) > 100: yaw_hist.pop(0)

        # ── DRAW ──────────────────────────────────
        screen.fill(BG)

        # Title bar
        pygame.draw.rect(screen, (25,25,40), (0,0,SW,48))
        screen.blit(ft.render("Car IMU Visualizer  |  MPU-9250  |  Bosch Future Mobility", True, WHITE),(15,13))
        screen.blit(fs.render("R = Reset", True, GRAY),(SW-110,16))

        # ── TOP VIEW ──
        panel(screen, (15,55,330,370), "TOP VIEW  —  YAW", fl)
        car_top(screen, 180, 250, yaw, fs)
        # compass
        ccx, ccy = 305, 115
        pygame.draw.circle(screen, GRAY, (ccx,ccy), 30, 1)
        for lbl,ang in [("N",0),("E",90),("S",180),("W",270)]:
            txt_c(screen, lbl, fs, GRAY,
                  int(ccx+38*math.sin(math.radians(ang))),
                  int(ccy-38*math.cos(math.radians(ang))))
        nx2 = ccx+24*math.sin(math.radians(yaw))
        ny2 = ccy-24*math.cos(math.radians(yaw))
        pygame.draw.line(screen, RED,(ccx,ccy),(int(nx2),int(ny2)),3)
        pygame.draw.circle(screen, RED,(ccx,ccy),4)

        # ── FRONT VIEW ──
        panel(screen, (360,55,330,370), "FRONT VIEW  —  ROLL", fl)
        car_front(screen, 525, 240, roll, fs)

        # ── SIDE VIEW ──
        panel(screen, (705,55,380,370), "SIDE VIEW  —  PITCH", fl)
        car_side(screen, 895, 235, -pitch, fs)

        # ── DATA PANEL ──
        panel(screen, (15,440,1070,245), "LIVE DATA", fl)

        def gauge(lbl, val, lo, hi, x, y, col):
            w = 220
            p = max(0.0, min(1.0, (val-lo)/(hi-lo)))
            pygame.draw.rect(screen, GRAY,  (x,y,w,18), border_radius=4)
            pygame.draw.rect(screen, col,   (x,y,int(w*p),18), border_radius=4)
            pygame.draw.rect(screen, WHITE, (x,y,w,18),1,border_radius=4)
            screen.blit(fd.render(f"{lbl}: {val:+7.2f} deg", True, WHITE),(x+w+10,y))

        gauge("ROLL ", roll,  -90,  90, 35, 490, CAR_BODY)
        gauge("PITCH", pitch, -90,  90, 35, 525, ARROW)
        gauge("YAW  ", yaw,  -180, 180, 35, 560, YELLOW)

        # yaw graph
        gx0,gy0,gw,gh = 330,455,400,120
        pygame.draw.rect(screen,(20,20,35),(gx0,gy0,gw,gh))
        pygame.draw.rect(screen,GRAY,(gx0,gy0,gw,gh),1)
        my = gy0+gh//2
        pygame.draw.line(screen,GRID,(gx0,my),(gx0+gw,my),1)
        screen.blit(fs.render("Yaw History",True,GRAY),(gx0+5,gy0+3))
        if len(yaw_hist) > 1:
            pts = [(gx0+int(i/100*gw), my-int(v/180*(gh//2-5)))
                   for i,v in enumerate(yaw_hist)]
            pygame.draw.lines(screen, YELLOW, False, pts, 2)

        # numeric
        screen.blit(fl.render("ORIENTATION",True,ACCENT),(760,460))
        for i,(lbl,val,col) in enumerate([("Roll  ",roll,CAR_BODY),
                                           ("Pitch ",pitch,ARROW),
                                           ("Yaw   ",yaw,YELLOW)]):
            screen.blit(fd.render(f"{lbl}: {val:+8.2f} deg",True,col),(760,490+i*28))

        screen.blit(fs.render("● LIVE",True,ARROW),(SW-70,SH-22))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()