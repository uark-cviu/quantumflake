# QuantumFlake: A Framework for 2D Material Detection and Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**QuantumFlake** is a streamlined framework for the automated detection and layer classification of 2D materials (flakes) in microscopy images. It provides a complete, easy-to-use pipeline from raw image to structured analysis, leveraging a YOLOv8 model for flake detection and a custom ResNet-based model for layer classification.

The framework is designed for accessibility and extensibility, allowing researchers to rapidly analyze images, train custom models on their own data, and integrate the tools into larger workflows.

You can test out the framework with your own images here: https://huggingface.co/spaces/sanpdy/quantum-flake-pipeline

---

## Key Features

- **One-Liner Inference & Training:** Run complex analysis or train new models from your terminal with single, simple commands.
- **Flexible & Data-Driven:** The pipeline automatically adapts to models with different architectures and class types.
- **Configuration-Driven:** Control everything—from model weights to hyperparameters—via YAML files for reproducible experiments.
- **Advanced Inference Options:** Includes patch-based inference for high-resolution images and color calibration for consistent results across different microscopes.
- **Extensible by Design:** The modular structure makes it easy to swap out models or add new functionality.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/sanpdy/quantumflake](https://github.com/sanpdy/quantumflake)
    cd quantumflake
    ```

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment (e.g., conda, venv).

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights:**
    Create a `weights` directory and place the pre-trained model files inside.
    ```bash
    mkdir weights
    ```
    You can download the official weights (trained on Masubuchi et al.'s GMMDetector dataset) from Hugging Face:
    - **Detector:** [yolo-flake-detector-GMM.pt](https://huggingface.co/sanpdy/yolo-flake-detector)
    - **Classifier:** [flake-classifier.pth](https://huggingface.co/sanpdy/flake-classifier)

## Quickstart: Running Inference

The primary way to interact with the framework is via the Command-Line Interface (CLI).

**1. Basic Prediction:**
Run detection and classification on a directory of images.

```bash
python -m quantumflake.cli predict "path/to/your/images/"
```

**2. Overriding Configuration:**
Use the `--opts` flag to change any setting on the fly. For example, to run on an Apple Silicon GPU and save visualizations:

```bash
python -m quantumflake.cli predict "data/your_image.jpg" --opts device=cpu save_vis=True
```

**3. Using a Custom Config File:**
For full control, specify your own YAML config file.

```bash
python -m quantumflake.cli predict "data/your_image.jpg" -c "path/to/your/config.yaml"
```

---

## Training New Models

QuantumFlake provides built-in, CLI-driven training workflows for both the detector and the classifier.

### Training a New Detector

The detector uses the standard **YOLO format**.

**1. Prepare Your Dataset:**

- Create a dataset directory with `images` and `labels` subfolders, split into `train` and `val`.
- For each image, create a `.txt` label file. Each line should be `0 <x_center> <y_center> <width> <height>` for the "flake" class.

**2. Create a Dataset YAML:**
Create a `your_dataset.yaml` file that points to your data.

```yaml
train: /path/to/your/detector_dataset/images/train
val: /path/to/your/detector_dataset/images/val
nc: 1
names: ["flake"]
```

**3. Run the Training Command:**

```bash
python -m quantumflake.cli train detector \
    --data "path/to/your/your_dataset.yaml" \
    --epochs 100 \
    --device 0  # Use GPU 0
```

The best model will be saved to `runs/detect/your_dataset/weights/best.pt`.

### Training a New Classifier

The classifier uses the standard **ImageFolder format**.

**1. Prepare Your Dataset:**

- Create a dataset directory. Inside, create one subdirectory for each class you want to train. The folder name becomes the class label.
- Place your cropped flake images into the corresponding class folders.

```
my_classifier_dataset/
├── monolayer/
│   ├── flake_01.png
│   └── ...
└── bilayer/
    └── ...
```

**2. Run the Training Command:**
The trainer will automatically discover the classes from your folder names.

```bash
python -m quantumflake.cli train classifier \
    --data "path/to/my_classifier_dataset" \
    --epochs 25 \
    --device cpu \
    --save-dir "runs/my_new_classifier" \
    --num-materials 2 \
    --material-dim 16
```

The best model will be saved to `runs/my_new_classifier/best_classifier.pth`. The pipeline will automatically read the class names and architecture from this file during inference.

---

## Advanced Configuration

Control the pipeline's behavior via a YAML configuration file.

**Example `config.yaml`:**

```yaml
# Hardware and output settings
device: "cpu"
output_dir: "monolayer_classifier_v2/"
save_vis: true

# --- Advanced Features ---

# Enable color calibration by providing a reference image path
calibration_ref_path: "/path/to/your/calibration_ref.png"

# Explicitly disable calibration (it's on by default if a path is provided)
use_calibration: false

# Enable patch-based inference for high-res images
patching:
  use_patching: true
  patch_size: 640

models:
  detector:
    weights: "weights/uark_detector_v3.pt"
    conf_thresh: 0.2
    iou_thresh: 0.05

  classifier:
    weights: "weights/flake_monolayer_classifier.pth"
    # For our current/older models, you MUST specify the architecture details
    class_names: ["1-layer", "5plus-layer"]
    num_materials: 2
    material_dim: 64
```

## Project Structure

```
quantumflake/
│
├─ quantumflake/
│  ├─ __init__.py
│  ├─ pipeline.py
│  ├─ cli.py
│  ├─ models/
│  ├─ trainers/
│  │  ├─ detect.py
│  │  └─ classify.py
│  ├─ utils/
│  └─ cfg/
│
├─ weights/
├─ requirements.txt
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
    <tr>
      <td>Xuan-Bac Nguyen</td>
      <td>University of Arkansas</td>
    </tr>
    <tr>
      <td>Sankalp Pandey</td>
      <td>University of Arkansas</td>
    </tr>
    <tr>
        <td>Tim Faltermeier</td>
        <td>Montana State University</td>
    </tr>
    <tr>
      <td>Dr. Hugh Churchill</td>
      <td>University of Arkansas</td>
    </tr>
    <tr>
      <td>Dr. Nicholas Borys</td>
      <td>Montana State University</td>
    </tr>
    <tr>
      <td>Dr. Khoa Luu</td>
      <td>University of Arkansas</td>
    </tr>
  </tbody>
</table>
</div>
