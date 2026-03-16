# DETR (HuggingFace Transformers)

## Training Support

QuantumFlake supports both DETR training and inference.

## Architecture

- Use`facebook/detr-resnet-50` for training.
- We do not currently publish a fine-tuned DETR flake checkpoint.

## Inference

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=detr \
         models.detector.detr.architecture=/path/to/detr_checkpoint_dir \
         device=cuda:0 save_vis=true output_dir=runs/predict_detr
```

Use a local checkpoint directory that was saved with both `model.save_pretrained(...)` and `processor.save_pretrained(...)`.

## Training

```bash
python -m quantumflake.cli train detector \
  --type detr \
  --architecture facebook/detr-resnet-50 \
  --train-json /path/to/train.json \
  --train-images /path/to/train_images \
  --val-json /path/to/val.json \
  --val-images /path/to/val_images \
  --num-labels 1 \
  --device cpu \
  --output-dir runs/detect_train/detr
```

Training expects COCO-format boxes and writes a Hugging Face checkpoint directory that can be used directly for inference.
