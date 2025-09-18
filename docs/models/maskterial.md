# MaskTerial (Detectron2 + Mask2Former)

## Requirements
- Detectron2 (pinned)
- Mask2Former CUDA op: `MultiScaleDeformableAttention`

## Weights & Config
```
weights/maskterial/
├── config.yaml
└── maskterial.pth
```

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=maskterial \
         models.detector.maskterial.architecture=weights/maskterial/config.yaml \
         models.detector.maskterial.weights=weights/maskterial/maskterial.pth \
         models.detector.conf_thresh=0.01 \
         device=cuda:0 save_vis=true output_dir=runs/predict_maskterial
```

> If you see `No object named 'MaskFormer'`, verify that the CUDA op is installed.
