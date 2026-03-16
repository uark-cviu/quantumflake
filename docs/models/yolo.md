# YOLO (Ultralytics)

## Weights

- The standard public detector checkpoint is `uark_detector_v3.pt`.
- Place it anywhere you want and pass the path with `models.detector.yolo.weights=...`.

## Inference

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=yolo \
         models.detector.yolo.weights=weights/uark_detector_v3.pt \
         device=cuda:0 save_vis=true output_dir=runs/predict_yolo
```

## Training

```bash
python -m quantumflake.cli train detector \
  --type yolo \
  --data /path/to/dataset.yaml \
  --weights /path/to/init_detector.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0 \
  --output-dir runs/detect_train
```

Expected dataset YAML:

```yaml
train: /path/to/images/train
val: /path/to/images/val
nc: 1
names: ["flake"]
```

Use a `detect` checkpoint as `--weights`. A segmentation checkpoint will fail against a detect-format dataset.
