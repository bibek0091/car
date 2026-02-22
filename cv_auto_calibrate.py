import cv2
import numpy as np

print("================================================")
print("       AUTO-CALIBRATE TRAFFIC LIGHT HSV         ")
print("================================================")
print("INSTRUCTIONS:")
print("1. Point your webcam at the traffic light.")
print("2. When the Red or Green LED turns ON, click directly on the glowing bulb.")
print("3. The script will mathematically calculate the perfect HSV range for you.")
print("4. Press 'Q' to quit anytime.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

current_frame = None
current_hsv = None

def get_hsv_at_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_hsv is not None:
            # Get the exact HSV value at the clicked pixel
            h, s, v = current_hsv[y, x]
            
            print("\n-------------------------------------------")
            print(f"🎯 CLICKED PIXEL: Hue={h}, Sat={s}, Val={v}")
            print("-------------------------------------------")
            
            # Mathematically calculate a generous window around that exact color
            # Hue wraps around at 180 in OpenCV
            h_min = max(0, h - 15)
            h_max = min(179, h + 15)
            
            # We want anything semi-colorful to very colorful
            s_min = max(30, s - 60)
            s_max = 255
            
            # We want glowing things (high value/brightness), but allow some flex
            v_min = max(100, v - 80)
            v_max = 255
            
            print(f"✅ AUTO-CALCULATED RANGES FOR bfmc_pilot_v3_yolo.py:")
            print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
            print("-------------------------------------------\n")
            
            # Show a temporary mask of what this new range looks like
            mask = cv2.inRange(current_hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
            cv2.imshow("Auto-Calibrated Mask (Press any key to close)", mask)
            cv2.waitKey(0)

cv2.namedWindow('Webcam', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Webcam', get_hsv_at_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    current_frame = frame
    current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Draw a small crosshair in the middle to help aiming
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 1)

    cv2.putText(frame, "CLICK ON THE GLOWING LED BULB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
