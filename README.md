# QuantumFlake: A Framework for 2D Material Detection and Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**QuantumFlake** is a streamlined framework for the automated detection and layer classification of 2D quantum materials (flakes) in microscopy images. It provides a complete, easy-to-use pipeline from raw image to structured analysis, leveraging a YOLOv11 model for flake detection and a custom ResNet-based model for layer classification.

The framework is designed for accessibility and extensibility, allowing researchers to rapidly analyze microscopy images and integrate custom models with minimal setup.

You can test out the framework with your own images here: https://huggingface.co/spaces/sanpdy/quantum-flake-pipeline

---

## Key Features

- **One-Liner Inference:** Detect and classify all flakes in an image or directory using a single, simple command.
- **Powerful & Flexible CLI:** A robust command-line interface allows for easy control over all inference parameters.
- **Configuration-Driven:** Control everything, from model weights to confidence thresholds, via YAML files for reproducible and organized experiments.
- **Material-Aware Architecture:** The classifier is designed to incorporate material-specific information (though currently trained for image-only use), making it future-proof for more advanced models.
- **Placeholder Stubs:** The framework includes placeholder files and a clear structure for future expansion into training, model exporting, and quantization.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/sanpdy/quantumflake
    ```

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment (e.g., conda, venv).

    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Pre-trained Weights:**
    Create a `weights` directory in the project root and place the pre-trained model files inside it.

    ```bash
    mkdir weights
    ```

    Your directory should look like this:

    ```
    quantumflake/
    weights/
    ├── yolo-flake-detector.pt
    └── flake-classifier.pth
    ...
    ```

    You can download the pre-trained weights (trained on Masubuchi et al.'s GMMDetector dataset) from Hugging Face:

- **Detector weights** (`yolo-flake.pt`):  
  https://huggingface.co/sanpdy/yolo-flake-detector

- **Classifier weights** (`flake-classifier.pth`):  
  https://huggingface.co/sanpdy/flake-classifier

## Quickstart: Running Inference

Once installed, you can immediately start analyzing images.

### Using the Command-Line Interface (CLI)

The CLI is the primary way to interact with the framework.

**1. Basic Prediction:**
Run detection and classification on a single image using the default settings.

```bash
python -m quantumflake.cli predict "path/to/your/image.png"
```

**2. Overriding Configuration:**
Use the `--opts` flag to change any setting from the default configuration file on the fly. For example, to run on the CPU and save a visualization:

```bash
python -m quantumflake.cli predict "data/your_image.jpg" --opts device=cpu save_vis=True
```

A new annotated image will be saved to `runs/predict/vis_your_image.jpg`.

**3. Using a Custom Config File:**
For more complex experiments, you can specify your own YAML config file.

```bash
python -m quantumflake.cli predict "data/your_image.jpg" -c "quantumflake/cfg/my_gpu.yaml"
```

### Programmatic Usage (in Python)

You can easily integrate the pipeline into your own scripts.

```python
# example_script.py
from quantumflake import FlakePipeline
from quantumflake.utils.io import load_config
import pprint

# 1. Load the configuration
config = load_config("quantumflake/cfg/default.yaml")

# 2. Override settings if needed
config['device'] = 'cpu'
config['save_vis'] = True

# 3. Initialize and run the pipeline
pipeline = FlakePipeline(config)
results = pipeline("path/to/your/image.png")

# 4. Print the results
pprint.pprint(results)
```

## Configuration

The framework's behavior is controlled by YAML files located in `quantumflake/cfg/`.

- **`default.yaml`**: This file contains all the default parameters for the models and pipeline.
- **Custom Configs**: You can create new `.yaml` files (e.g., `my_gpu_experiment.yaml`) to define different experimental setups. The CLI will intelligently load your custom config on top of the default.

**Example `default.yaml` snippet:**

```yaml
device: "auto"
save_vis: false

models:
  detector:
    weights: "weights/yolo_flake.pt"
    conf_thresh: 0.25
    iou_thresh: 0.45

  classifier:
    weights: "weights/resnet18_flake.pth"
    num_materials: 4
    material_dim: 64
```

## Project Structure

```
quantumflake/
│
├─ quantumflake/          # The core library package
│  ├─ __init__.py
│  ├─ pipeline.py         # Main FlakePipeline class
│  ├─ cli.py              # Command-line interface logic
│  ├─ models/             # Model definitions
│  ├─ utils/              # Helper functions (IO, visualization)
│  └─ cfg/                # Default YAML configuration files
│
├─ weights/               # Default directory for model weights
├─ scripts/               # (Future) Standalone scripts for export, etc.
├─ tests/                 # (Future) Automated tests
├─ requirements.txt
└─ README.md              # This file
```

## Roadmap (Future Work)

The core inference pipeline is stable. Future development will focus on:

- **Training Scripts:** Implementing flexible training loops for both the detector and classifier.
- **Dataset Tools:** Adding utilities for converting annotations (e.g., from COCO) and splitting datasets.
- **Model Export & Quantization:** Providing scripts to export models to ONNX and apply quantization for faster CPU inference.
- **Automated Testing:** Building a full test suite with `pytest`.

## Contributing

If you have suggestions, please open an issue to discuss it, or feel free to email me. Thanks!
