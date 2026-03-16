# MaskTerial (Detectron2 + Mask2Former)

## Training Support

QuantumFlake does not currently provide a dedicated `train detector --type maskterial` command. Use upstream MaskTerial training to produce weights, then use QuantumFlake for inference with those artifacts.

## Requirements
- Detectron2 (pinned)
- `MultiScaleDeformableAttention` for Mask2Former-based configs

## Weights & Config
QuantumFlake does not currently publish a MaskTerial flake checkpoint in the public `uark-cviu` Hugging Face org. Supply your own compatible config + weights.

- MRCNN-style YAML configs can be loaded directly.
- Mask2Former configs require the `MultiScaleDeformableAttention` op in addition to Detectron2.

## Inference
```bash
python -m quantumflake.cli predict "/path/to/images" \
  --opts models.detector.type=maskterial \
         models.detector.maskterial.architecture=/path/to/config.yaml \
         models.detector.maskterial.weights=/path/to/maskterial.pth \
         models.detector.conf_thresh=0.01 \
         device=cuda:0 save_vis=true output_dir=runs/predict_maskterial
```

You can also point `architecture` at `maskterial://...` after the bootstrap code has downloaded the upstream repository into the cache directory.

> If you see `No object named 'MaskFormer'`, verify that the Mask2Former CUDA op is installed and that you are using a Mask2Former-based config.
