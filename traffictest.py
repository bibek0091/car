import cv2
import time
import os
from ultralytics import YOLO

# -------------------------------------------------------
# STEP 1: Find and load the model
# -------------------------------------------------------
possible_paths = [
    "best_traffic_med_yolo_v8.pt",
    "best_traffic_med_yolo_v8/weights/best.pt",
    "best_traffic_med_yolo_v8",
    "weights/best_traffic_med_yolo_v8.pt",
]

model = None
loaded_path = None

for path in possible_paths:
    if os.path.exists(path):
        print(f"Found model at: {path}")
        try:
            model = YOLO(path)
            loaded_path = path
            break
        except Exception as e:
            print(f"Failed to load from {path}: {e}")

if model is None:
    print("ERROR: Could not find model in any expected location")
    print("Files in current directory:")
    for f in os.listdir("."):
        print(f"  {f}")
    exit()

print(f"Model loaded from: {loaded_path}")
print(f"Model classes: {model.names}")
print(f"Number of classes: {len(model.names)}")

# -------------------------------------------------------
# STEP 2: Camera setup
# -------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera opened. Press Q to quit.")
print("Press + to increase sensitivity, - to decrease")

# -------------------------------------------------------
# STEP 3: Settings
# -------------------------------------------------------
CONF_THRESHOLD = 0.15       # Very low to catch everything
PROCESS_EVERY_N = 3         # Process every 3rd frame
frame_count = 0
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    frame_count += 1

    # -------------------------------------------------------
    # STEP 4: Run model every N frames
    # -------------------------------------------------------
    if frame_count % PROCESS_EVERY_N == 0:
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        last_detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                last_detections.append((x1, y1, x2, y2, conf, label))
                # Print every detection to terminal
                print(f"DETECTED -> Label: {label} | Conf: {conf:.3f} | Box: {x1},{y1},{x2},{y2}")

    # -------------------------------------------------------
    # STEP 5: Draw detections
    # -------------------------------------------------------
    detected_any = len(last_detections) > 0

    for (x1, y1, x2, y2, conf, label) in last_detections:
        label_lower = label.lower()

        if "red" in label_lower:
            box_color = (0, 0, 255)
            status = "RED - STOP"
        elif "green" in label_lower:
            box_color = (0, 255, 0)
            status = "GREEN - GO"
        elif "yellow" in label_lower or "orange" in label_lower:
            box_color = (0, 200, 255)
            status = "YELLOW - SLOW"
        else:
            box_color = (255, 255, 0)
            status = f"UNKNOWN: {label}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        # Label tag
        tag = f"{status} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # -------------------------------------------------------
    # STEP 6: Big top status bar
    # -------------------------------------------------------
    if detected_any:
        top_label = last_detections[0][5].lower()
        if "red" in top_label:
            bar_color = (0, 0, 200)
            bar_text = ">>> DETECTED: RED LIGHT - STOP <<<"
        elif "green" in top_label:
            bar_color = (0, 160, 0)
            bar_text = ">>> DETECTED: GREEN LIGHT - GO <<<"
        elif "yellow" in top_label:
            bar_color = (0, 160, 220)
            bar_text = ">>> DETECTED: YELLOW LIGHT - SLOW <<<"
        else:
            bar_color = (80, 80, 80)
            bar_text = f">>> DETECTED: {last_detections[0][5].upper()} <<<"

        cv2.rectangle(frame, (0, 0), (640, 55), bar_color, -1)
        cv2.putText(frame, bar_text, (10, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (640, 55), (30, 30, 30), -1)
        cv2.putText(frame, "NO TRAFFIC LIGHT DETECTED",
                    (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 80, 80), 2, cv2.LINE_AA)

    # -------------------------------------------------------
    # STEP 7: Bottom info bar
    # -------------------------------------------------------
    cv2.rectangle(frame, (0, 455), (640, 480), (20, 20, 20), -1)
    cv2.putText(frame,
                f"Conf: {CONF_THRESHOLD:.2f} | Frame: {frame_count} | Classes: {list(model.names.values())}",
                (5, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow("Traffic Light Model Test", frame)

    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("+"):
        CONF_THRESHOLD = max(0.05, round(CONF_THRESHOLD - 0.05, 2))
        print(f"Confidence threshold lowered to: {CONF_THRESHOLD}")
    elif key == ord("-"):
        CONF_THRESHOLD = min(0.95, round(CONF_THRESHOLD + 0.05, 2))
        print(f"Confidence threshold raised to: {CONF_THRESHOLD}")

cap.release()
cv2.destroyAllWindows()