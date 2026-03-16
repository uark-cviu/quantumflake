# Model Zoo

This page links to the per-backend docs with configs, checkpoints, and CLI examples.

Training support summary:

- YOLO: train + infer
- DETR: train + infer
- ViTDet: train + infer
- OpenVINO-YOLO: inference only, export from YOLO first
- MaskTerial: inference only in QuantumFlake, train with upstream MaskTerial
- ResNet classifier: train + infer

- [YOLO (Ultralytics)](models/yolo.md)
- [DETR (HF Transformers)](models/detr.md)
- [ViTDet (Detectron2)](models/vitdet.md)
- [OpenVINO-YOLO (CPU)](models/openvino_yolo.md)
- [MaskTerial (Mask2Former)](models/maskterial.md)
- [ResNet Layer Classifier](models/classifier.md)

> Tip: Place your weights under `weights/` as described in each page.
