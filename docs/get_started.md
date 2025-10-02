# Getting Started

## Installation (CUDA 11.8 example)

```bash
conda create -n quantumflake python=3.12 -y
conda activate quantumflake

# PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118

# Core deps
pip install ultralytics transformers timm opencv-python pillow tqdm pyyaml yacs scikit-image numba

# Detectron2 (pin a known-good commit)
pip install git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761

# Mask2Former CUDA op (for MaskTerial)
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  MultiScaleDeformableAttention==1.0+9b0651cpt2.5.1cu118
```

## Quick Inference

```bash
python -m quantumflake.cli predict "/path/to/images_or_glob"
```

See the [Model Zoo](model_zoo.md) for per-backend examples.
