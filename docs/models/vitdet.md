# ViTDet (Detectron2)

## Training Support

QuantumFlake supports both ViTDet training and inference.

## Bootstrap

QuantumFlake bootstraps the ViTDet project into the cache directory on first run. Set `QUANTUMFLAKE_CACHE_DIR=/path/to/cache` if you want to override the default cache location.

## Weights

QuantumFlake does not currently publish a ViTDet flake checkpoint. Supply a compatible Detectron2 checkpoint such as `model_final.pth` from a previous ViTDet training run.

## Inference

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=vitdet \
         models.detector.vitdet.architecture="vitdet://COCO/mask_rcnn_vitdet_b_100ep.py" \
         models.detector.vitdet.weights=/path/to/model_final.pth \
         device=cuda:0 save_vis=true output_dir=runs/predict_vitdet
```

The ViTDet backend is box-only inside QuantumFlake. If your COCO training set has boxes but no `segmentation` entries, the training path automatically disables the mask branch.

## Training

```bash
python -m quantumflake.cli train detector \
  --type vitdet \
  --vit-arch vitdet://COCO/mask_rcnn_vitdet_b_100ep.py \
  --train-json /path/to/train.json \
  --train-images /path/to/train_images \
  --val-json /path/to/val.json \
  --val-images /path/to/val_images \
  --device cpu \
  --ims-per-batch 1 \
  --output-dir runs/detect_train/vitdet
```

Training writes a Detectron2 checkpoint such as `model_final.pth`, which you then feed back into the inference command.
