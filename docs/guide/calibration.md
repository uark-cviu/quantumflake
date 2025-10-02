# Color Calibration

Set a reference image and enable calibration in your config:

```yaml
use_calibration: true
calibration_ref_path: "assets/calibration_ref.jpg"
```

The pipeline will normalize colors prior to detection/classification.
