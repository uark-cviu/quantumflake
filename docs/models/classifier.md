# ResNet Layer Classifier

A lightweight classifier that predicts layer categories (e.g., *1-layer*, *5plus-layer*) on chip crops.

## Training
```bash
python -m quantumflake.cli train classifier \
  --data my_dataset \
  --epochs 25 \
  --device cuda:0 \
  --save-dir runs/classify \
  --num-materials 2 \
  --material-dim 64
```

## Expected Classes
```yaml
classifier:
  class_names: ["1-layer", "5plus-layer"]
```
