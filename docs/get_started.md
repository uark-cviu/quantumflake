# Getting Started

## Recommended Installation

Create an environment with Python 3.11 or 3.12, install a matching PyTorch build for your hardware, then install QuantumFlake in editable mode:

```bash
conda create -n quantumflake python=3.11 -y
conda activate quantumflake

# Install the PyTorch build that matches your platform:
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Install QuantumFlake and its core runtime dependencies
pip install -e .
```

Notes:

- The standard `predict` flow only needs the core package, a detector checkpoint, and a classifier checkpoint.
- Optional backends require extra packages:
  - ViTDet / MaskTerial: Detectron2
  - MaskTerial: `MultiScaleDeformableAttention`
  - OpenVINO-YOLO: `openvino`
- ViTDet and MaskTerial bootstrap upstream source trees into a cache directory on first use. Set `QUANTUMFLAKE_CACHE_DIR=/path/to/cache` if you want to override the default cache location.

## Download Checkpoints

Download the files needed from [2D Quantum Material Characterization collection](https://huggingface.co/collections/uark-cviu/2d-quantum-material-characterization) and either:

- create a local `weights/` folder and place them there, or
- point the CLI directly to absolute paths with `--opts`

The public `uark-cviu` Hugging Face org currently publishes:

- `uark-cviu/flake-detector`
- `uark-cviu/flake-classifier`

For the standard YOLO + classifier pipeline, you need:

- `uark_detector_v3.pt`
- `flake_monolayer_classifier.pth`

Other detector backends need user-supplied compatible artifacts:

- DETR: a fine-tuned checkpoint directory saved with `save_pretrained(...)`
- ViTDet: a Detectron2 `model_final.pth` checkpoint
- MaskTerial: your own config + weights
- OpenVINO-YOLO: an exported OpenVINO IR (`.xml` + `.bin`) from a YOLO `.pt` checkpoint

For optional domain adaptation usage from $\varphi$-Adapt, download `spectrum_inv.pth`.

## Quick Inference

```bash
python -m quantumflake.cli predict "/path/to/images_or_glob" \
  --opts models.detector.type=yolo \
         models.detector.yolo.weights=weights/uark_detector_v3.pt \
         models.classifier.weights=weights/flake_monolayer_classifier.pth \
         device=cpu \
         output_dir=runs/predict
```

This writes:

- `vis_<image_name>` overlays when `save_vis=true`
- `<image_stem>.json` in `output_dir`

See the [Model Zoo](model_zoo.md) for backend-specific examples.
