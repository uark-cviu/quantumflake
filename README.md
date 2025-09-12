
# QuantumFlake: 2D Flake Detection & Layer Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**QuantumFlake** is a modular framework for automated **detection** and **layer classification** of 2D-material flakes in microscope images. It provides a one-command pipeline:

<h3 align="center">detect → crop → classify → visualize</h3>

Supported detectors:
- **YOLO (Ultralytics)**
- **DETR (HuggingFace Transformers)**
- **ViTDet (Detectron2)**
- **OpenVINO-YOLO (CPU)**
- **MaskTerial (Detectron2 + Mask2Former)**


## Highlights
- One-command inference on folders or glob patterns  
- Standardized output across detectors  
- ResNet-based classifier  
- Optional color calibration, patch-based inference, JSON sidecars  

## Installation
Example (CUDA 11.8):

```bash
conda create -n quantumflake python=3.12 -y
conda activate quantumflake

# PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118

# Core deps
pip install ultralytics transformers timm opencv-python pillow tqdm pyyaml yacs scikit-image numba

# Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761

# Mask2Former CUDA op
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  MultiScaleDeformableAttention==1.0+9b0651cpt2.5.1cu118
```

## Weights

Organize weights in a `weights/` folder:

* YOLO detector: `weights/uark_detector_v3.pt`
* Classifier: `weights/flake_monolayer_classifier.pth`

Other backends (DETR, ViTDet, OpenVINO-YOLO, MaskTerial) accept their standard configs/checkpoints.

>**Note:** The first time you use **ViTDet** or **MaskTerial**, QuantumFlake will auto-download the minimal upstream code into `~/.cache/quantumflake/` (no full repo clone needed).

## Inference

Default config:

```bash
python -m quantumflake.cli predict "/path/to/images_or_glob"
```

Override options with `--opts`.

**YOLO:**

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=yolo \
         models.detector.yolo.weights=weights/uark_detector_v3.pt \
         device=cuda:0 \
         save_vis=true \
         output_dir=runs/predict_yolo
```

**DETR:**

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=detr \
         models.detector.detr.architecture=facebook/detr-resnet-50 \
         device=cuda:0 \
         save_vis=true \
         output_dir=runs/predict_detr
```
> For custom DETR heads, set `models.detector.num_labels` if your checkpoint expects a different class count.

**ViTDet:**

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=vitdet \
         models.detector.vitdet.architecture="vitdet://COCO/mask_rcnn_vitdet_b_100ep.py" \
         device=cuda:0 \
         save_vis=true \
         output_dir=runs/predict_vitdet
```

**OpenVINO-YOLO (CPU):**

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=openvino_yolo \
         models.detector.openvino_yolo.weights=weights/yolo_openvino/model.xml \
         device=cpu \
         save_vis=true \
         output_dir=runs/predict_openvino
```

**MaskTerial (Detectron2 + Mask2Former):**

```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=maskterial \
         models.detector.maskterial.architecture=weights/maskterial/config.yaml \
         models.detector.maskterial.weights=weights/maskterial/maskterial.pth \
         models.detector.conf_thresh=0.01 \
         device=cuda:0 \
         save_vis=true \
         output_dir=runs/predict_maskterial
```

> If you see `No object named 'MaskFormer'` or MSDeformAttn warnings, ensure
> `MultiScaleDeformableAttention`, `scikit-image`, and `numba` are installed (see Installation).

## Pipeline

1. Detect flakes with the selected backend
2. Crop detections to flake chips
3. Classify crops (e.g., 1-layer, 5+ layers)
4. Save visualizations and JSON sidecars

Example JSON record:

```json
{
  "bbox": [x1, y1, x2, y2],
  "det_conf": 0.8731,
  "cls": "1-layer",
  "cls_conf": 0.9123
}
```

Overlays are saved as `vis_<image>.png`, and per-image detections as `<image_stem>.json` inside `output_dir`.

## Configuration

Use `-c config.yaml` or override with `--opts`.

Example:

```yaml
device: "cuda:0"
output_dir: "runs/predict"
save_vis: true

# Optional color calibration (off unless path is provided)
use_calibration: false
calibration_ref_path: ""  # e.g., "assets/calibration_ref.jpg"

# Optional patch-based inference for large images
patching:
  use_patching: false
  patch_size: 640

models:
  detector:
    type: "yolo"
    conf_thresh: 0.20
    iou_thresh: 0.05
    yolo:
      weights: "weights/uark_detector_v3.pt"

  classifier:
    weights: "weights/flake_monolayer_classifier.pth"
    class_names: ["1-layer", "5plus-layer"]
    num_materials: 2
    material_dim: 64
```

## Training

### Detector (YOLO)

Dataset YAML:

```yaml
train: /path/to/detector_dataset/images/train
val:   /path/to/detector_dataset/images/val
nc: 1
names: ["flake"]
```

Train:

```bash
python -m quantumflake.cli train detector \
  --data /path/to/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --device 0
```

### Classifier (ImageFolder)

Folder structure:

```
my_dataset/
├── 1-layer/
│   ├── flake_01.png
│   └── ...
└── 5plus-layer/
    └── ...
```

Train:

```bash
python -m quantumflake.cli train classifier \
  --data my_dataset \
  --epochs 25 \
  --device cuda:0 \
  --save-dir runs/classify \
  --num-materials 2 \
  --material-dim 64
```

## Extending

All detectors must return:

* `boxes.xyxy → Tensor[N,4]`
* `boxes.conf → Tensor[N]`
* `orig_img → np.ndarray (BGR)`

Add your backend in `models/detector.py` and expose config keys.

---

## Troubleshooting

* **MaskTerial error**: install `MultiScaleDeformableAttention` wheel.
* **Blank predictions**: lower `models.detector.conf_thresh`.
* **Large images**: enable patching.
* **OpenVINO issues**: weights must point to `.xml`.


## Project Structure

```
quantumflake/
│
├─ quantumflake/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ pipeline.py
│  ├─ cfg/
│  │   └─ default.yaml
│  ├─ models/
│  │   └─ detector.py
│  ├─ trainers/
│  │   ├─ detect.py
│  │   └─ classify.py
│  └─ utils/
│      ├─ io.py
│      ├─ data.py
│      ├─ vis.py
│      ├─ vitdet_bootstrap.py
│      ├─ maskterial_bootstrap.py
│      └─ m2f_bootstrap.py
│
├─ weights/
│   ├─ uark_detector_v3.pt
│   ├─ flake_monolayer_classifier.pth
│   └─ maskterial/
│       ├─ config.yaml
│       └─ maskterial.pth
└─ README.md
```

<h2 align="center">Contributors</h2>
<div align="center">

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Affiliation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Xuan-Bac Nguyen</td><td>University of Arkansas</td></tr>
    <tr><td>Sankalp Pandey</td><td>University of Arkansas</td></tr>
    <tr><td>Tim Faltermeier</td><td>Montana State University</td></tr>
    <tr><td>Dr. Hugh Churchill</td><td>University of Arkansas</td></tr>
    <tr><td>Dr. Nicholas Borys</td><td>Montana State University</td></tr>
    <tr><td>Dr. Khoa Luu</td><td>University of Arkansas</td></tr>
  </tbody>
</table>
</div>
