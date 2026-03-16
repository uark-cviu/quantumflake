# OpenVINO-YOLO (CPU)

## Training Support

QuantumFlake does not train OpenVINO models directly. This backend is inference-only and expects an OpenVINO IR exported from a YOLO `.pt` checkpoint.

## Export

QuantumFlake does not publish a prebuilt OpenVINO IR. Export one from a YOLO `.pt` checkpoint first:

```bash
yolo export model=/path/to/flake_detector.pt format=openvino imgsz=640
```

Then point QuantumFlake at the generated `.xml` file (the sibling `.bin` file is loaded automatically).

## Inference

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=openvino_yolo \
         models.detector.openvino_yolo.weights=/path/to/yolo_openvino/model.xml \
         device=cpu save_vis=true output_dir=runs/predict_openvino
```

Note: x86 CPU environments are the safer target for this backend.
