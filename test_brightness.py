import cv2
import time
import numpy as np
from ultralytics import YOLO

print("================================================")
print("  DIFFERENTIAL BRIGHTNESS TRAFFIC LIGHT TEST    ")
print("================================================")
print("Loading Best YOLO Model...")

try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam Active. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection for Bosch objects
    results = model.predict(frame, conf=0.3, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # --- DIFFERENTIAL BRIGHTNESS LOGIC ---
            if label == "traffic-light":
                # Ensure valid coordinates
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                
                box_h = cy2 - cy1
                box_w = cx2 - cx1
                
                if box_h > 10 and box_w > 5:
                    # 1. Crop to the black housing
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                    
                    # 2. Convert to Grayscale (Ignore Color Washout)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
                    # Blur slightly to merge the LED pixels together
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # 3. Find the absolute brightest pixel value in the box
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
                    
                    # Threshold everything that is close to the max brightness (e.g. top 10%)
                    _, bright_mask = cv2.threshold(blurred, max(200, max_val - 30), 255, cv2.THRESH_BINARY)
                    
                    # 4. Find the center of mass (centroid) of the brightest pixels
                    M = cv2.moments(bright_mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # 5. Check if the bright mass is in the Top or Bottom half
                        midpoint_y = box_h / 2.0
                        
                        status = "GREEN (GO)"
                        color = (0, 255, 0)
                        if cY < midpoint_y:
                            status = "RED (STOP)"
                            color = (0, 0, 255)
                            
                        # Draw standard YOLO box
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
                        
                        # Draw the bright pixel centroid
                        cv2.circle(frame, (cx1 + cX, cy1 + cY), 5, (255, 255, 255), -1)
                        
                        # Draw Midpoint Line to visualize the split
                        cv2.line(frame, (cx1, cy1 + int(midpoint_y)), (cx2, cy1 + int(midpoint_y)), (255, 255, 0), 1)
                        
                        # Draw Status Tag
                        tag = f"TL: {status}"
                        cv2.putText(frame, tag, (cx1, max(20, cy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Show the actual mask algorithm to the user as an overlay inset
                        mask_bgr = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
                        # Resize inset so it's easy to see
                        inset_h, inset_w = 100, int(100 * (box_w/box_h))
                        inset = cv2.resize(mask_bgr, (inset_w, inset_h))
                        # Place inset in top left corner of main frame
                        frame[0:inset_h, 0:inset_w] = inset
                        cv2.rectangle(frame, (0, 0), (inset_w, inset_h), (255, 255, 0), 1)
                        cv2.putText(frame, "Mask Inset", (0, inset_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Draw other Bosch objects normally just so we know YOLO is running
            elif label in ["stop-sign", "car", "pedestrian", "no-entry-road-sign"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("Differential Brightness Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
