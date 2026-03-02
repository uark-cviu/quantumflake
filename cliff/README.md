# CLIFF: Continual Learning for Incremental Flake Features in 2D Material Identification

This repository presents the official code for **CLIFF**, a continual learning framework for material-incremental 2D flake classification.

<h3 align="center">Abstract</h3>

Identifying quantum flakes is crucial for scalable quantum hardware; however, automated layer classification from optical microscopy remains challenging due to substantial appearance shifts across different materials. In this paper, we propose a new Continual-Learning Framework for Flake Layer Classification (CLIFF). To our knowledge, this is the first systematic study of continual learning in the domain of two-dimensional (2D) materials. Our method enables the model to differentiate between materials and their physical and optical properties by freezing a backbone and base head trained on a reference material. For each new material, it learns a material-specific prompt, embedding, and a delta head. A prompt pool and a cosine-similarity gate modulate features and compute material-specific corrections. Additionally, we incorporate memory replay with knowledge distillation. CLIFF achieves competitive accuracy with significantly lower forgetting than naive fine-tuning and a prompt-based baseline.


## Citation

```bibtex
@misc{pandey2026cliffcontinuallearningincremental,
      title={CLIFF: Continual Learning for Incremental Flake Features in 2D Material Identification}, 
      author={Sankalp Pandey and Xuan Bac Nguyen and Nicholas Borys and Hugh Churchill and Khoa Luu},
      year={2026},
      eprint={2508.17261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.17261}, 
}
```
