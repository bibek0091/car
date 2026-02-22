import cv2
import numpy as np

def nothing(x):
    pass

# Initialize OpenCV Window and Sliders
cv2.namedWindow("Traffic Light HSV Tuner")
cv2.resizeWindow("Traffic Light HSV Tuner", 640, 300)

cv2.createTrackbar("H Min", "Traffic Light HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("S Min", "Traffic Light HSV Tuner", 50, 255, nothing)
cv2.createTrackbar("V Min", "Traffic Light HSV Tuner", 150, 255, nothing)

cv2.createTrackbar("H Max", "Traffic Light HSV Tuner", 10, 179, nothing)
cv2.createTrackbar("S Max", "Traffic Light HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("V Max", "Traffic Light HSV Tuner", 255, 255, nothing)

# Defaults matching bfmc_pilot
print("--- OpenCV Traffic Light Tuning Script ---")
print("1. Point your webcam at the traffic light on the track.")
print("2. Adjust the 6 sliders until ONLY the glowing LED bulb is pure white in the [Mask] window, and everything else is black.")
print("3. Press 'SPACE' to print the exact numbers to your terminal.")
print("4. Press 'Q' to quit.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

cv2.namedWindow('Webcam', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get current slider values
    h_min = cv2.getTrackbarPos("H Min", "Traffic Light HSV Tuner")
    s_min = cv2.getTrackbarPos("S Min", "Traffic Light HSV Tuner")
    v_min = cv2.getTrackbarPos("V Min", "Traffic Light HSV Tuner")
    
    h_max = cv2.getTrackbarPos("H Max", "Traffic Light HSV Tuner")
    s_max = cv2.getTrackbarPos("S Max", "Traffic Light HSV Tuner")
    v_max = cv2.getTrackbarPos("V Max", "Traffic Light HSV Tuner")
    
    # Pre-process image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # Generate Mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Calculate screen coverage to show "Activation" percent
    h, w = mask.shape
    total_pixels = h * w
    active_pixels = cv2.countNonZero(mask)
    ratio = (active_pixels / total_pixels) * 100
    
    # HUD
    cv2.putText(frame, f"Activation Ratio: {ratio:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Tune until ONLY the LED bulb is white in the mask.", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Webcam', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Filtered Result', res)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        print(f"\n[PERFECT MATCH LOGGED]")
        print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])\n")

cap.release()
cv2.destroyAllWindows()
