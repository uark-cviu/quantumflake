import os
import json
import glob
from pathlib import Path
from PIL import Image

def yolo_to_coco(images_dir, labels_dir, out_json, class_name="flake"):
    images = []
    annotations = []
    categories = [{"id": 1, "name": class_name}]
    ann_id = 1
    img_id = 1

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        img_paths += glob.glob(os.path.join(images_dir, "**", ext), recursive=True)

    for ip in sorted(img_paths):
        try:
            w, h = Image.open(ip).size
        except Exception:
            continue
        images.append({"id": img_id, "file_name": os.path.relpath(ip, images_dir), "width": w, "height": h})

        label_path = os.path.join(labels_dir, Path(ip).with_suffix(".txt").name)
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (cx - bw/2.0) * w
                    y = (cy - bh/2.0) * h
                    ww = bw * w
                    hh = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x, y, ww, hh],
                        "area": ww*hh,
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"COCO annotations written to {out_json} (images={len(images)}, anns={len(annotations)})")
