import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import json

from .models.detector import get_detector, StandardizedResults
from .models.classifier import FlakeLayerClassifier
from .utils.data import crop_flakes, load_image
from .utils.vis import draw_overlay
from .utils.io import resolve_path
from phi_adapt.modules import ShiftModule

class FlakePipeline:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(self.cfg['device'])
        print(f"Initializing pipeline on device: {self.device}")
        self.color_ref_bgr = None
        use_calibration = self.cfg.get('use_calibration', False)
        if use_calibration:
            ref_path_str = self.cfg.get('calibration_ref_path')
            if ref_path_str:
                ref_path = resolve_path(ref_path_str)
                self.color_ref_bgr = cv2.imread(str(ref_path))
                if self.color_ref_bgr is None:
                    print(f"WARNING: Color reference image not found at {ref_path}. Calibration will be skipped.")
                else:
                    print(f"Color calibration reference loaded from: {ref_path}")

        use_phi_adapt = self.cfg['models'].get('use_phi_adapt', False)
        if use_phi_adapt:
            n_wavelengths = self.cfg['models']['n_wavelengths']
            min_wavelength = self.cfg['models']['min_wavelength']
            max_wavelength = self.cfg['models']['max_wavelength']
            self.shift_module = ShiftModule(min_wavelength, max_wavelength, n_wavelengths)
        else:
            self.shift_module = None

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.detector = get_detector(self.cfg)
        cls_weights_path = resolve_path(self.cfg['models']['classifier']['weights'])
        cls_params = self.cfg['models']['classifier']

        checkpoint = torch.load(cls_weights_path, map_location=self.device)
        self.class_names = checkpoint.get('class_names') or cls_params.get('class_names')
        if not self.class_names:
            raise ValueError("Class names not found in classifier checkpoint or config.")
        num_classes = checkpoint.get('num_classes', len(self.class_names))
        num_materials = checkpoint.get('num_materials', cls_params.get('num_materials'))
        material_dim = checkpoint.get('material_dim', cls_params.get('material_dim'))

        print(f"Loading classifier with {num_classes} classes: {self.class_names}")
        print(f"Model architecture: num_materials={num_materials}, material_dim={material_dim}")

        self.classifier = FlakeLayerClassifier(
            num_materials=num_materials,
            material_dim=material_dim,
            num_classes=num_classes,
        )

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.classifier.load_state_dict(state_dict, strict=False)
        self.classifier.to(self.device).eval()

        print("Pipeline initialized successfully.")

    def __call__(self, image_source, save_vis=None):
        should_save_vis = save_vis if save_vis is not None else self.cfg.get('save_vis', False)
        if isinstance(image_source, (list, tuple)):
            return [self._process_single(src, should_save_vis) for src in tqdm(image_source, desc="Processing Images")]
        else:
            return self._process_single(image_source, should_save_vis)

    def _run_detection(self, image_bgr: np.ndarray):
        patching_cfg = self.cfg.get('patching', {})
        use_patching = patching_cfg.get('use_patching', False)

        det_params = self.cfg['models']['detector']
        conf = float(det_params.get('conf_thresh', 0.20))
        iou = float(det_params.get('iou_thresh', 0.05))

        if not use_patching:
            print("Running detection on full image...")
            res = self.detector.predict(image_bgr, conf=conf, iou=iou)
            if isinstance(res, list):
                res = res[0]
            return res

        print("Running detection on patches (batched)...")
        patch_size = int(patching_cfg.get('patch_size', 640))
        h, w, _ = image_bgr.shape

        patches, patch_coords = [], []
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image_bgr[y:y+patch_size, x:x+patch_size]
                if patch.size > 0:
                    patches.append(patch)
                    patch_coords.append((x, y))

        if not patches:
            return None

        all_det_results = self.detector.predict(patches, conf=conf, iou=iou)

        all_boxes_xyxy, all_confs = [], []
        for i, det_result in enumerate(all_det_results):
            offset_x, offset_y = patch_coords[i]
            boxes_tensor = det_result.boxes.xyxy if hasattr(det_result.boxes, "xyxy") else torch.empty((0, 4))
            conf_tensor = det_result.boxes.conf if hasattr(det_result.boxes, "conf") else torch.empty((0,))
            for j in range(len(boxes_tensor)):
                x1, y1, x2, y2 = boxes_tensor[j].tolist()
                all_boxes_xyxy.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])
                all_confs.append(float(conf_tensor[j].item()) if len(conf_tensor) > j else 0.0)

        if not all_boxes_xyxy:
            return None

        return StandardizedResults(all_boxes_xyxy, all_confs, image_bgr)

    def _process_single(self, image_path, save_vis: bool):
        try:
            orig_bgr = load_image(image_path)

            proc_bgr = orig_bgr
            calibration_active = False
            if self.color_ref_bgr is not None:
                proc_bgr = self.calibration(self.color_ref_bgr, orig_bgr)
                calibration_active = True
            
            if self.shift_module is not None:
                # print("Applying physics-based domain adaptation...")
                proc_bgr = self.shift_module(proc_bgr)

            det_results = self._run_detection(proc_bgr)
            if det_results is None or len(det_results.boxes) == 0:
                return []

            boxes = det_results.boxes
            crops_pil = crop_flakes(det_results.orig_img, boxes.xyxy.tolist())
            if not crops_pil:
                return []

            processed_crops = torch.stack([self.preprocess(c) for c in crops_pil]).to(self.device)
            with torch.no_grad():
                logits = self.classifier(processed_crops)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                cls_confs, cls_indices = torch.max(probabilities, 1)

            final_results = self._package_results(boxes, cls_indices.cpu(), cls_confs.cpu())

            if save_vis:
                output_dir = resolve_path(self.cfg.get('output_dir', 'runs/predict'))
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = Path(image_path).name
                output_path = output_dir / f"vis_{fname}"

                image_to_draw_on = proc_bgr if calibration_active else orig_bgr
                draw_overlay(image_to_draw_on, final_results, str(output_path))
                with open(output_dir / f"{Path(image_path).stem}.json", "w") as f:
                    json.dump(final_results, f, indent=2)

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
            target_lab[:, :, i] = ((target_lab[:, :, i] - tgt_mean) * (src_std / tgt_std) + src_mean).clip(0, 255)
        return cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR).astype(np.uint8)

    def _package_results(self, det_boxes, cls_indices, cls_confs):
        results = []
        for i in range(len(det_boxes)):
            results.append({
                "bbox": [int(round(c)) for c in det_boxes.xyxy[i].cpu().numpy().tolist()],
                "det_conf": round(float(det_boxes.conf[i].cpu().item()), 4) if len(det_boxes.conf) > i else None,
                "cls": self.class_names[int(cls_indices[i])],
                "cls_conf": round(float(cls_confs[i].item()), 4),
            })
        return results
