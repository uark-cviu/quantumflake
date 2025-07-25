from ultralytics import YOLO

def load_detector(weights_path: str, device: str):
    """
    Loads the YOLOv11 object detection model.

    Args:
        weights_path (str): Path to the .pt model weights file.
        device (str): Device to load the model onto ('cpu', 'cuda:0', etc.).

    Returns:
        A YOLO model instance.
    """
    print(f"Loading detector from: {weights_path}")
    model = YOLO(weights_path)
    model.to(device)
    return model