import cv2
from PIL import Image
import numpy as np

def load_image(image_path):
    """Loads an image using OpenCV and converts to BGR."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

def crop_flakes(image_bgr: np.ndarray, bboxes_xyxy: list) -> list:
    """
    Crops detected flakes from the main image.

    Args:
        image_bgr (np.ndarray): The full source image in BGR format.
        bboxes_xyxy (list): A list of bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        list: A list of cropped images as PIL.Image objects.
    """
    crops = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return crops
