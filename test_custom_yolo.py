import os
import cv2
from ultralytics import YOLO

def test_model():
    model_path = "runs/detect/custom_toy_traffic/weights/best.pt"
    print(f"Loading {model_path}...")
    model = YOLO(model_path)
    
    # Get a list of the class names in the model
    print("Model Class Names:", model.names)
    
    # Pick a random image from the training set to verify detection
    img_dir = "traffic/train/images"
    test_img = None
    for f in os.listdir(img_dir):
        if f.endswith(".jpg"):
            test_img = os.path.join(img_dir, f)
            break
            
    if test_img:
        print(f"Testing on image: {test_img}")
        results = model.predict(test_img, conf=0.1)
        for box in results[0].boxes:
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            print(f"DETECTED -> Label: '{label}' | Confidence: {conf:.2f} | Bbox: {box.xyxy[0].tolist()}")
    else:
        print("No test image found.")

if __name__ == "__main__":
    test_model()
