# CLIFF: Continual Learning for Incremental Flake Features in 2D Material Identification

This repository presents the official code for **CLIFF**, a continual learning framework for material-incremental 2D flake classification.

<h3 align="center">Abstract</h3>

Identifying quantum flakes is crucial for scalable quantum hardware; however, automated layer classification from optical microscopy remains challenging due to substantial appearance shifts across different materials. This paper proposes a new Continual-Learning Framework for Flake Layer Classification (CLIFF). To the best of our knowledge, this work represents the first systematic study of continual learning in two-dimensional (2D) materials. The proposed framework enables the model to distinguish materials and their physical and optical properties by freezing the backbone and base head, which are trained on a reference material. For each new material, it learns a material-specific prompt, embedding, and a delta head. A prompt pool and a cosine-similarity gate modulate features and compute material-specific corrections. Additionally, memory replay with knowledge distillation is incorporated. CLIFF achieves competitive accuracy with significantly lower forgetting than naive fine-tuning and a prompt-based baseline.


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
