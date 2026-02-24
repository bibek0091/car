import os
from ultralytics import YOLO

def train_new_model():
    print("==================================================")
    print("   Training Custom YOLO on BFMC.v1-001.yolo26 Data")
    print("==================================================")
    
    # Paths
    base_model_path = "yolo26n.pt"  # Assumes we run this from inside bfmc_deploy
    data_yaml_path = "c:/Users/p23mi/Downloads/archive/BFMC.v1-001.yolo26/data.yaml"
    
    if not os.path.exists(base_model_path):
        print(f"[ERROR] Base model '{base_model_path}' not found in current directory.")
        return
        
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] Dataset config '{data_yaml_path}' not found.")
        return

    print(f"Loading base model: {base_model_path}")
    model = YOLO(base_model_path)
    
    print(f"Starting training with dataset: {data_yaml_path}")
    print("This will take some time...")
    
    # Train the model
    # Adjust epochs and batch size as needed based on hardware
    results = model.train(
        data=data_yaml_path,
        epochs=10,             # Keep it small for test turnaround, increase for production
        imgsz=640,
        batch=16,
        project="runs/detect",   
        name="bfmc_v1_yolo26",   # Name of the output folder containing the weights
        device=0                 # 0 for GPU, 'cpu' for CPU
    )
    
    print("==================================================")
    print("   Training Complete!")
    print("   New weights saved to: runs/detect/bfmc_v1_yolo26/weights/best.pt")
    print("==================================================")

if __name__ == "__main__":
    train_new_model()
