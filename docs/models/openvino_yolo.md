# OpenVINO-YOLO (CPU)

## Weights
Point to your OpenVINO IR:
- `weights/yolo_openvino/model.xml` (and `.bin`)

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=openvino_yolo \
         models.detector.openvino_yolo.weights=weights/yolo_openvino/model.xml \
         device=cpu save_vis=true output_dir=runs/predict_openvino
```
