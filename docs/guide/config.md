# Configuration System

Use `-c config.yaml` or override with `--opts`:

```bash
python -m quantumflake.cli predict "images/*" \
  --opts models.detector.type=yolo models.detector.yolo.weights=weights/uark_detector_v3.pt
```

Example schema:

```yaml
device: "cuda:0"
output_dir: "runs/predict"
save_vis: true

patching:
  use_patching: false
  patch_size: 640

models:
  detector:
    type: "yolo"
    conf_thresh: 0.20
    iou_thresh: 0.05
    yolo:
      weights: "weights/uark_detector_v3.pt"
  classifier:
    weights: "weights/flake_monolayer_classifier.pth"
    class_names: ["1-layer", "5plus-layer"]
```
