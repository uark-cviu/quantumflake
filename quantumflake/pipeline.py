# quantumflake/pipeline.py

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .models.detector import load_detector
from .models.classifier import FlakeLayerClassifier, CLASS_NAMES
from .utils.data import crop_flakes
from .utils.vis import draw_overlay
from .utils.io import resolve_path

class FlakePipeline:
    """
    Orchestrates flake detection and classification.
    Corrected to match the exact training conditions.
    """
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(self.cfg['device'])
        print(f"Initializing pipeline on device: {self.device}")

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # -------------------------------------------------------------------------

        # --- Load Models ---
        det_weights = resolve_path(self.cfg['models']['detector']['weights'])
        cls_weights = resolve_path(self.cfg['models']['classifier']['weights'])
        cls_params = self.cfg['models']['classifier']

        self.detector = load_detector(det_weights, self.device)
        self.classifier = FlakeLayerClassifier(
            num_materials=cls_params['num_materials'],
            material_dim=cls_params['material_dim'],
            num_classes=len(CLASS_NAMES),
            freeze_cnn=cls_params.get('freeze_cnn', False)
        )
        checkpoint = torch.load(cls_weights, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()

        print("Pipeline initialized successfully.")

    def __call__(self, image_source, save_vis=None):
        """
        Processes images. The `material_id` argument is removed as it's not used.
        """
        should_save_vis = save_vis if save_vis is not None else self.cfg.get('save_vis', False)

        if isinstance(image_source, (list, tuple)):
            return [self._process_single(src, should_save_vis) for src in tqdm(image_source, desc="Processing Images")]
        else:
            return self._process_single(image_source, should_save_vis)

    def _process_single(self, image_path, save_vis):
        """Helper to process one image."""
        try:
            det_params = self.cfg['models']['detector']
            det_results = self.detector.predict(
                source=image_path, conf=det_params['conf_thresh'], iou=det_params['iou_thresh'], verbose=False
            )[0]

            img_bgr = det_results.orig_img
            boxes = det_results.boxes
            if not len(boxes): return []

            crops_pil = crop_flakes(img_bgr, boxes.xyxy)
            if not crops_pil: return []
            
            # Preprocess the batch of cropped images using our defined transforms
            processed_crops = torch.stack([self.preprocess(c) for c in crops_pil]).to(self.device)

            with torch.no_grad():
                # Call the model exactly as it was called during training (no material_id)
                logits = self.classifier(processed_crops)
                
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                cls_confs, cls_indices = torch.max(probabilities, 1)

            final_results = self._package_results(boxes, cls_indices.cpu(), cls_confs.cpu())
            if save_vis:
                output_dir = resolve_path(self.cfg.get('output_dir', 'runs/predict'))
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = Path(image_path).name if isinstance(image_path, (str, Path)) else "vis_output.png"
                output_path = output_dir / f"vis_{fname}"
                draw_overlay(img_bgr, final_results, str(output_path))
                print(f"Visualization saved to: {output_path}")

            return final_results
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise

    def _package_results(self, det_boxes, cls_indices, cls_confs):
        results = []
        for i in range(len(det_boxes)):
            results.append({
                "bbox": [round(c) for c in det_boxes.xyxy[i].cpu().numpy().tolist()],
                "det_conf": round(float(det_boxes.conf[i].cpu().item()), 4),
                "cls": CLASS_NAMES[cls_indices[i]],
                "cls_conf": round(float(cls_confs[i].item()), 4),
            })
        return results
