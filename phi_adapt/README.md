# φ-Adapt: A Physics-Informed Adaptation Learning Approach to 2D Quantum Material Discovery

This repository presents the official code for **φ-Adapt**, a physics-informed adaptation learning approach to 2D flake identification.

<h3 align="center">Abstract</h3>

Characterizing quantum flakes is a critical step in quantum hardware engineering because the quality of these flakes directly influences qubit performance. Although computer vision methods for identifying two-dimensional quantum flakes have emerged, they still face significant challenges in estimating flake thickness. These challenges include limited data, poor generalization, sensitivity to domain shifts, and a lack of physical interpretability. In this paper, we introduce one of the first Physics-informed Adaptation Learning approaches to overcome these obstacles. We focus on two main issues, i.e., data scarcity and generalization. First, we propose a new synthetic data generation framework that produces diverse quantum flake samples across various materials and configurations, reducing the need for time-consuming manual collection. Second, we present -Adapt, a physics-informed adaptation method that bridges the performance gap between models trained on synthetic data and those deployed in real-world settings. Experimental results show that our approach achieves state-of-the-art performance on multiple benchmarks, outperforming existing methods. Our proposed approach advances the integration of physics-based modeling and domain adaptation. It also addresses a critical gap in leveraging synthesized data for real-world 2D material analysis, offering impactful tools for deep learning and materials science communities.

## Instruction

### Installation

Please follow the installation in [here](https://github.com/uark-cviu/quantumflake?tab=readme-ov-file#installation).

### Module Weights

Organize the weights in a `weights/` folder as:
- `weights/spectrum_inv.pth`

### How to use

The module can work with every detection models. To enable **φ-Adapt**, run:
```bash
python -m quantumflake.cli predict "/path/to/images_or_glob" \
    --opts models.use_phi_adapt=true \
           models.spectrum_inv.weights=weights/spectrum_inv.pth
```

## Citation

```bibtex
@article{nguyen2025varphi,
  title={$$\backslash$varphi $-Adapt: A Physics-Informed Adaptation Learning Approach to 2D Quantum Material Discovery},
  author={Nguyen, Hoang-Quan and Nguyen, Xuan Bac and Pandey, Sankalp and Faltermeier, Tim and Borys, Nicholas and Churchill, Hugh and Luu, Khoa},
  journal={arXiv preprint arXiv:2507.05184},
  year={2025}
}
```
