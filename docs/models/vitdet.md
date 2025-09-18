# ViTDet (Detectron2)

## Bootstrap
QuantumFlake bootstraps minimal ViTDet code into `~/.cache/quantumflake/` on first run.

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=vitdet \
         models.detector.vitdet.architecture="vitdet://COCO/mask_rcnn_vitdet_b_100ep.py" \
         device=cuda:0 save_vis=true output_dir=runs/predict_vitdet
```
