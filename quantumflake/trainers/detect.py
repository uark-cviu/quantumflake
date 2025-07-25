from ultralytics import YOLO
from pathlib import Path
import yaml

def train(data, epochs=50, imgsz=640, device="0", weights="yolo11n.pt", project="runs/detect"):
    """
    Args:
        data (str): Path to the dataset's YAML configuration file. This file
                    should define the dataset paths and class names.
        epochs (int): The total number of training epochs.
        imgsz (int): The image size (height and width) for training.
        device (str): The device to train on. Can be a single GPU ('0'),
                      multiple GPUs ('0,1'), or 'cpu'.
        weights (str): Path to the initial model weights. Can be an official
                       YOLOv8 model (e.g., 'yolov8n.pt') to train from scratch
                       or a path to your own checkpoint to fine-tune.
        project (str): The root directory where training runs will be saved.
    """
    print("\n--- Starting Detector Training ---")
    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found at: {data}")

    with open(data_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        num_classes = len(data_yaml.get('names', []))
        if num_classes != 1:
            print(f"WARNING: Expected 1 class for flake detection, but found {num_classes} in {data}. Proceeding anyway.")

    model = YOLO(weights)
    print(f"Starting training on device '{device}' with dataset '{data}'...")
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=project,
        name=data_path.stem
    )

    print("\n--- Detector Training Finished ---")
    print(f"Results and best model saved in '{project}/{data_path.stem}'")