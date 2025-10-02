import cv2
import hashlib

CLASS_COLORS = {
    '1-layer': (255, 56, 56),    # Red
    '2-layer': (56, 176, 255),   # Blue
    '3-layer': (87, 255, 56),    # Green
    '4-layer': (255, 255, 0),    # Yellow
    '5plus-layer': (180, 130, 255),
}
DEFAULT_COLOR = (200, 200, 200)

def _color_for_label(label: str):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = int(hashlib.sha1(label.encode()).hexdigest(), 16)
    return (50 + (h % 206), 50 + (h // 256 % 206), 50 + (h // 65536 % 206))

def draw_overlay(image_bgr, results, output_path=None):
    img_out = image_bgr.copy()
    for r in results:
        bbox = r['bbox']
        cls_name = r['cls']
        label = f"{cls_name}: {r['cls_conf']:.2f}" if r.get('cls_conf') is not None else f"{cls_name}"
        color = _color_for_label(cls_name)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(img_out, (x1, label_y - h - 5), (x1 + w, label_y), color, -1)
        cv2.putText(img_out, label, (x1, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if output_path:
        cv2.imwrite(str(output_path), img_out)
    return img_out
