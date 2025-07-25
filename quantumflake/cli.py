import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import json
from .models.detector import load_detector
from .models.classifier import FlakeLayerClassifier
from .utils.data import crop_flakes, load_image
from .utils.vis import draw_overlay
from .utils.io import resolve_path

class FlakePipeline:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(self.cfg['device'])
        print(f"Initializing pipeline on device: {self.device}")

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        det_weights = resolve_path(self.cfg['models']['detector']['weights'])
        self.detector = load_detector(det_weights, self.device)

        cls_weights_path = resolve_path(self.cfg['models']['classifier']['weights'])
        cls_params = self.cfg['models']['classifier']
        
        checkpoint = torch.load(cls_weights_path, map_location=self.device)
        self.class_names = checkpoint.get('class_names')
        num_classes = checkpoint.get('num_classes')
        num_materials = checkpoint.get('num_materials')
        material_dim = checkpoint.get('material_dim')

        if self.class_names is None or num_classes is None:
            print("WARNING: 'class_names' or 'num_classes' not found in checkpoint. Using values from config file.")
            self.class_names = cls_params['class_names']
            num_classes = len(self.class_names)
        
        if num_materials is None or material_dim is None:
            print("WARNING: 'num_materials' or 'material_dim' not found in checkpoint. Using values from config file.")
            num_materials = cls_params['num_materials']
            material_dim = cls_params['material_dim']
        
        print(f"Loading classifier with {num_classes} classes: {self.class_names}")
        print(f"Model architecture: num_materials={num_materials}, material_dim={material_dim}")

        self.classifier = FlakeLayerClassifier(
            num_materials=num_materials,
            material_dim=material_dim,
            num_classes=num_classes,
        )
        
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: 'model_state_dict' key not found. Assuming checkpoint is the state dict itself.")
            self.classifier.load_state_dict(checkpoint)

        self.classifier.to(self.device)
        self.classifier.eval()

        print("Pipeline initialized successfully.")

    def __call__(self, image_source, save_vis=None):
        should_save_vis = save_vis if save_vis is not None else self.cfg.get('save_vis', False)
        if isinstance(image_source, (list, tuple)):
            return [self._process_single(src, should_save_vis) for src in tqdm(image_source, desc="Processing Images")]
        else:
            return self._process_single(image_source, should_save_vis)

    def _process_single(self, image_path, save_vis):
        try:
            det_params = self.cfg['models']['detector']
            det_results = self.detector.predict(
                source=image_path, conf=det_params['conf_thresh'], iou=det_params['iou_thresh'], verbose=False
            )[0]

            orig_bgr = det_results.orig_img.copy()
            boxes = det_results.boxes
            if not len(boxes): return []

            crops_pil = crop_flakes(orig_bgr, boxes.xyxy)
            if not crops_pil: return []

            processed_crops = torch.stack([self.preprocess(c) for c in crops_pil]).to(self.device)

            with torch.no_grad():
                logits = self.classifier(processed_crops)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                cls_confs, cls_indices = torch.max(probabilities, 1)

            final_results = self._package_results(boxes, cls_indices.cpu(), cls_confs.cpu())

            if save_vis:
                output_dir = resolve_path(self.cfg.get('output_dir', 'runs/predict'))
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = Path(image_path).name if isinstance(image_path, (str, Path)) else "vis_output.png"
                output_path = output_dir / f"vis_{fname}"
                draw_overlay(orig_bgr, final_results, str(output_path))
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
                "cls": self.class_names[cls_indices[i]],
                "cls_conf": round(float(cls_confs[i].item()), 4),
            })
        return results
