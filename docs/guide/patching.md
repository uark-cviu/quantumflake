# Patch-based Inference

Enable patching for very large microscope images:

```yaml
patching:
  use_patching: true
  patch_size: 640
```

This reduces memory pressure and avoids downscaling artifacts.
