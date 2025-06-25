# trainers/detect.py
import Path
from ultralytics import YOLO

def train(data, epochs=50, imgsz=640, device="0"):
    model = YOLO("weights/yolo_flake.pt")
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project="runs/detect",
        name=Path(data).stem,
    )
