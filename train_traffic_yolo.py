import os
from ultralytics import YOLO

def main():
    print("========================================")
    print("   TRAINING CUSTOM BFMC TRAFFIC YOLO    ")
    print("========================================")
    
    # 1. Load the blazing-fast Nano model
    model = YOLO("yolov8n.pt") 
    
    # 2. Get the absolute path to the dataset
    data_path = os.path.abspath("traffic/data.yaml")
    
    print(f"Dataset Path: {data_path}")
    print("Starting training: 100 Epochs (This is very fast for 53 images!)...")
    
    # 3. Train the model exclusively on the toy traffic lights
    model.train(
        data=data_path,
        epochs=100,      
        imgsz=384,       # The resolution exported by Roboflow
        batch=16,
        name="custom_toy_traffic"
    )
    
    print("\n========================================")
    print("TRAINING COMPLETE! \u2705")
    print("Your new highly-specialized weights are located at:")
    print("runs/detect/custom_toy_traffic/weights/best.pt")
    print("========================================")

if __name__ == "__main__":
    main()
