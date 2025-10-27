# CLIFF: Continual Learning for Incremental Flake Features in 2D Material Identification

This repository presents the official code for **CLIFF**, a continual learning framework for material-incremental 2D flake classification.

<h3 align="center">Abstract</h3>

Characterizing quantum flakes is a critical step in quantum hardware engineering because the quality of these flakes directly influences qubit performance. Although computer vision methods for identifying two-dimensional quantum flakes have emerged, they still face significant challenges in estimating flake thickness. These challenges include limited data, poor generalization, sensitivity to domain shifts, and a lack of physical interpretability. In this paper, we introduce one of the first Physics-informed Adaptation Learning approaches to overcome these obstacles. We focus on two main issues, i.e., data scarcity and generalization. First, we propose a new synthetic data generation framework that produces diverse quantum flake samples across various materials and configurations, reducing the need for time-consuming manual collection. Second, we present -Adapt, a physics-informed adaptation method that bridges the performance gap between models trained on synthetic data and those deployed in real-world settings. Experimental results show that our approach achieves state-of-the-art performance on multiple benchmarks, outperforming existing methods. Our proposed approach advances the integration of physics-based modeling and domain adaptation. It also addresses a critical gap in leveraging synthesized data for real-world 2D material analysis, offering impactful tools for deep learning and materials science communities.


## Citation

```bibtex
@misc{pandey2025cliff,
title={CLIFF: Continual Learning for Incremental Flake Features in 2D Material Identification},
author={Sankalp Pandey and Xuan Bac Nguyen and Nicholas Borys and Hugh Churchill and Khoa Luu},
year={2025},
eprint={2508.17261},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2508.17261},
}
```
