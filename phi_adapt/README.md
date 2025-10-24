# φ-Adapt: A Physics-Informed Adaptation Learning Approach to 2D Quantum Material Discovery

This repository presents the official code for **φ-Adapt**, a physics-informed adaptation learning approach to 2D flake identification.

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
