# YOLO (Ultralytics)

## Weights
- Put your checkpoint at `weights/uark_detector_v3.pt`

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=yolo \
         models.detector.yolo.weights=weights/uark_detector_v3.pt \
         device=cuda:0 save_vis=true output_dir=runs/predict_yolo
```
