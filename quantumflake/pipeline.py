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
        ref_path_str = self.cfg.get('calibration_ref_path')
        if ref_path_str:
            ref_path = resolve_path(ref_path_str)
            self.color_ref_bgr = cv2.imread(str(ref_path))
            if self.color_ref_bgr is None:
                print(f"WARNING: Color reference image not found at {ref_path}. Calibration will be skipped.")
            else:
                print(f"Color calibration reference loaded from: {ref_path}")
        else:
            self.color_ref_bgr = None

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        det_weights = resolve_path(self.cfg['models']['detector']['weights'])
        self.detector = load_detector(det_weights, self.device)

        cls_weights_path = resolve_path(self.cfg['models']['classifier']['weights'])
        cls_params = self.cfg['models']['classifier']
        
        self.class_names = cls_params['class_names']
        num_classes = len(self.class_names)
        num_materials = cls_params['num_materials']
        material_dim = cls_params['material_dim']
        
        print(f"Loading classifier with {num_classes} classes: {self.class_names}")
        print(f"Model architecture: num_materials={num_materials}, material_dim={material_dim}")

        self.classifier = FlakeLayerClassifier(
            num_materials=num_materials,
            material_dim=material_dim,
            num_classes=num_classes,
        )
        
        checkpoint = torch.load(cls_weights_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
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
            orig_bgr = load_image(image_path)
            proc_bgr = orig_bgr
            use_calibration = self.cfg.get('use_calibration', False)
            if self.color_ref_bgr is not None and use_calibration:
                print("Applying color calibration before processing...")
                proc_bgr = self.calibration(self.color_ref_bgr, orig_bgr)

            det_params = self.cfg['models']['detector']
            det_results = self.detector.predict(
                source=proc_bgr, conf=det_params['conf_thresh'], iou=det_params['iou_thresh'], verbose=False
            )[0]

            boxes = det_results.boxes
            if not len(boxes): return []
            crops_pil = crop_flakes(proc_bgr, boxes.xyxy)
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

    def calibration(self, source_img, target_img):
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

        for i in range(3):
            src_mean, src_std = cv2.meanStdDev(source_lab[:, :, i])
            tgt_mean, tgt_std = cv2.meanStdDev(target_lab[:, :, i])

            target_lab[:, :, i] = (
                (target_lab[:, :, i] - tgt_mean) * (src_std / tgt_std) + src_mean
            ).clip(0, 255)

        corrected_img = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
        return corrected_img.astype(np.uint8)

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
