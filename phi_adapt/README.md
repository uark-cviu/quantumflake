# φ-Adapt: A Physics-Informed Adaptation Learning Approach to 2D Quantum Material Discovery

This repository presents the official code for **φ-Adapt**, a physics-informed adaptation learning approach to 2D flake identification.

<h3 align="center">Abstract</h3>

Characterizing quantum flakes is a critical step in quantum hardware engineering because the quality of these flakes directly influences qubit performance. Although computer vision methods for identifying two-dimensional quantum flakes have emerged, they still face significant challenges in estimating flake thickness. These challenges include limited data, poor generalization, sensitivity to domain shifts, and a lack of physical interpretability. In this paper, we introduce one of the first Physics-informed Adaptation Learning approaches to overcome these obstacles. We focus on two main issues, i.e., data scarcity and generalization. First, we propose a new synthetic data generation framework that produces diverse quantum flake samples across various materials and configurations, reducing the need for time-consuming manual collection. Second, we present -Adapt, a physics-informed adaptation method that bridges the performance gap between models trained on synthetic data and those deployed in real-world settings. Experimental results show that our approach achieves state-of-the-art performance on multiple benchmarks, outperforming existing methods. Our proposed approach advances the integration of physics-based modeling and domain adaptation. It also addresses a critical gap in leveraging synthesized data for real-world 2D material analysis, offering impactful tools for deep learning and materials science communities.

## Instruction

### Installation

Please follow the installation in [here](https://github.com/uark-cviu/quantumflake?tab=readme-ov-file#installation).

### How to use

`φ-Adapt` is included here as a research subproject, but it is not part of the default QuantumFlake quick-start.

For the standard `quantumflake` inference pipeline, you only need:

- a detector checkpoint
- a classifier checkpoint

The top-level pipeline does not currently load an external `spectrum_inv.pth`, so that file should not be documented as a required artifact for normal inference.

Treat the in-tree `models.use_phi_adapt` path as experimental until the adaptation-specific checkpoint loading and preprocessing path are finalized.

## Citation

```bibtex
@article{nguyen2025varphi,
  title={$$\backslash$varphi $-Adapt: A Physics-Informed Adaptation Learning Approach to 2D Quantum Material Discovery},
  author={Nguyen, Hoang-Quan and Nguyen, Xuan Bac and Pandey, Sankalp and Faltermeier, Tim and Borys, Nicholas and Churchill, Hugh and Luu, Khoa},
  journal={arXiv preprint arXiv:2507.05184},
  year={2025}
}
```
