import cv2
import time
from ultralytics import YOLO

# 1. Load the three models
print("Loading Models... This might take a few seconds.")
try:
    model_nano = YOLO("best_traffic_nano_yolo.pt")
    model_small = YOLO("best_traffic_small_yolo.pt")
    model_med = YOLO("best_traffic_med_yolo_v8.pt")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

models = {
    "NANO (Fastest)": model_nano,
    "SMALL (Middle)": model_small,
    "MEDIUM (Heavy)": model_med
}

model_keys = list(models.keys())
current_idx = 0

# 2. Open Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

conf_thresh = 0.20
print("\n=== Controls ===")
print("Press 'T' to Toggle between Nano, Small, and Medium models.")
print("Press '+' to Increase Confidence.")
print("Press '-' to Decrease Confidence.")
print("Press 'Q' to Quit.\n")

frame_count = 0
last_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    frame_count += 1
    
    # Calculate FPS
    now = time.time()
    dt = now - last_time
    if dt >= 1.0:
        fps = frame_count / dt
        frame_count = 0
        last_time = now

    # Current Model
    model_name = model_keys[current_idx]
    active_model = models[model_name]

    # Run Inference
    start_infer = time.time()
    results = active_model.predict(frame, conf=conf_thresh, verbose=False)
    infer_time = (time.time() - start_infer) * 1000  # ms
    
    # Draw Detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = active_model.names[cls]
            
            # Draw Box
            color = (0, 255, 0)
            if "red" in label.lower():
                color = (0, 0, 255)
            elif "yellow" in label.lower() or "orange" in label.lower():
                color = (0, 200, 255)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Tag
            tag = f"{label} {conf:.2f}"
            cv2.putText(frame, tag, (x1, max(20, y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # --- Draw Dashboard ---
    # Top Bar Header
    cv2.rectangle(frame, (0, 0), (640, 40), (40, 40, 40), -1)
    cv2.putText(frame, f"Model: {model_name}", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
    # Bottom Bar Stats
    cv2.rectangle(frame, (0, 440), (640, 480), (20, 20, 20), -1)
    stats = f"Infer: {infer_time:.1f}ms | Loop FPS: {fps:.1f} | Conf: {conf_thresh:.2f}"
    cv2.putText(frame, stats, (10, 465), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Multi-Model YOLO Tester", frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        current_idx = (current_idx + 1) % len(model_keys)
        print(f"Switched model to: {model_keys[current_idx]}")
    elif key == ord('+'):
        conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
        print(f"Conf raised to {conf_thresh}")
    elif key == ord('-'):
        conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
        print(f"Conf lowered to {conf_thresh}")

cap.release()
cv2.destroyAllWindows()
