# DETR (HuggingFace Transformers)

## Default Architecture
- `facebook/detr-resnet-50`

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=detr \
         models.detector.detr.architecture=facebook/detr-resnet-50 \
         device=cuda:0 save_vis=true output_dir=runs/predict_detr
```

> If using a custom head, set `models.detector.num_labels` appropriately.
