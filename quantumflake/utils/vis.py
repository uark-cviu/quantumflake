import cv2
import numpy as np

CLASS_COLORS = {
    '1-layer': (255, 56, 56),   # Red
    '2-layer': (56, 176, 255),    # Blue
    '3-layer': (87, 255, 56),  # Green
    '4-layer': (255, 255, 0) # Yellow
}
DEFAULT_COLOR = (200, 200, 200)

def draw_overlay(image_bgr, results, output_path=None):
    """Draws bounding boxes and labels on an image."""
    img_out = image_bgr.copy()
    for r in results:
        bbox = r['bbox']
        label = f"{r['cls']}: {r['cls_conf']:.2f}"
        color = CLASS_COLORS.get(r['cls'], DEFAULT_COLOR)
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(img_out, (x1, label_y - h - 5), (x1 + w, label_y), color, -1)
        cv2.putText(img_out, label, (x1, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if output_path:
        cv2.imwrite(str(output_path), img_out)
    return img_out

