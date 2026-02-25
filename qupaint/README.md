# QuPAINT: Physics-Aware Instruction Tuning Approach to Quantum Material Discovery

This is the code for QuPAINT, a physics-aware multimodal framework for quantum material (2D flake) characterization from optical microscopy images. QuPAINT combines (i) physics-based synthetic generation, (ii) instruction supervision, and (iii) physics-informed attention to learn robust flake representations.

- [Project Page](https://uark-cviu.github.io/projects/qupaint/)
- [Paper](https://arxiv.org/abs/2602.17478)
- [Dataset](https://uark-my.sharepoint.com/:f:/g/personal/sankalpp_uark_edu/IgDiKftTT9A4QbiZJgZy--GZAYOqZmG4MGUe126NJJ44uao?e=G1iQcS)
- Code: TBD (to be released)

## QF-Bench Dataset

QF-Bench is a large-scale benchmark for quantum material characterization, covering multiple materials, substrates, and imaging settings. Currently, a small subset (50 images) of the QF-Bench Dataset is released.

| Material  |   Mono    |    Few    |    Thick    |    Total    |
| :-------: | :-------: | :-------: | :---------: | :---------: |
|    BN     |    13     |    111    |   10,100    |   10,224    |
| Graphene  |   1,856   |   2,081   |    4,270    |    8,207    |
|   MoS₂    |    246    |    536    |   108,352   |   109,134   |
|   MoSe₂   |    24     |    32     |    1,327    |    1,383    |
|  MoWSe₂   |     9     |    35     |     293     |     337     |
|    WS₂    |    43     |     5     |    2,144    |    2,192    |
|   WSe₂    |    207    |    35     |    6,195    |    6,437    |
|   WTe₂    |    148    |    582    |   141,882   |   142,612   |
| **Total** | **2,546** | **3,417** | **274,563** | **280,526** |

<sub><i>Breakdown of annotated flake counts for each material by layer (Mono, Few, Thick) in the full dataset.</i></sub>

### Directory structure

```
dataset/QF-Bench/
├── annotations.json
└── materials/
    ├── mos2/
    ├── mose2/
    ├── mowse2/
    ├── ws2/
    └── wse2/
```

### Annotations

The file `QF-Bench/annotations.json` follows the standard COCO detection format.

#### Categories

Each flake instance is assigned one of the following layer classes:

| ID  | Name  |
| --- | ----- |
| 1   | Mono  |
| 2   | Few   |
| 3   | Thick |

- **Mono**: Monolayer flakes
- **Few**: Few-layer flakes
- **Thick**: Thick or bulk-like flakes

