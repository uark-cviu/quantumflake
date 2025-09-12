import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForObjectDetection
_DETECTRON2_OK = True
try:
    from detectron2.config import LazyConfig, instantiate, get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.engine import DefaultPredictor
except Exception:
    _DETECTRON2_OK = False

class StandardizedResults:
    def __init__(self, boxes_xyxy: List[List[float]], confs: List[float], orig_img: np.ndarray):
        self.boxes = self._create_box_object(boxes_xyxy, confs)
        self.orig_img = orig_img

    class BoxObject:
        def __init__(self, xyxy, conf):
            self.xyxy = torch.tensor(xyxy, dtype=torch.float32) if len(xyxy) else torch.empty((0, 4), dtype=torch.float32)
            self.conf = torch.tensor(conf, dtype=torch.float32) if len(conf) else torch.empty((0,), dtype=torch.float32)
        def __len__(self):
            return len(self.xyxy)

    def _create_box_object(self, xyxy, conf):
        return self.BoxObject(xyxy, conf)

class YOLODetector:
    def __init__(self, model_path: str, device: str):
        print(f"[YOLO] Loading from: {model_path}")
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            pass
        self.device = device

    def predict(self, image: Union[np.ndarray, List[np.ndarray]], conf: float, iou: float):
        return self.model.predict(source=image, conf=conf, iou=iou, verbose=False)

class FastOpenVINOYOLO:
    def __init__(self, model_path: str, device: str):
        import openvino as ov
        print(f"[OpenVINO YOLO] Loading from: {model_path}")
        xml_path = model_path
        if not model_path.endswith('.xml'):
            xmls = list(Path(model_path).glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(f"No .xml file found in {model_path}")
            xml_path = str(xmls[0])

        core = ov.Core()
        self.model = core.read_model(xml_path)
        self.input_layer = self.model.input(0)
        self.input_height = int(self.input_layer.shape[2])
        self.input_width  = int(self.input_layer.shape[3])
        self.output_ports = list(self.model.outputs)
        print(f"[OpenVINO YOLO] Model input shape: {self.input_layer.shape}")
        self.compiled_model = core.compile_model(self.model, "CPU")
        print("[OpenVINO YOLO] Compiled for CPU")
        self._ov = ov

    def _preprocess_image(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        r = min(self.input_height / h, self.input_width / w)
        new_h, new_w = int(h * r), int(w * r)
        pad_h, pad_w = (self.input_height - new_h) // 2, (self.input_width - new_w) // 2

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded  = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]
        return input_tensor, r, (pad_w, pad_h)

    def _pick_detection_output(self, result: Dict[Any, np.ndarray]) -> np.ndarray:
        candidates = []
        for port in self.output_ports:
            arr = result[port]
            if arr.ndim == 3:
                candidates.append(arr)
        if not candidates:
            return result[self.output_ports[0]]
        candidates.sort(key=lambda a: (a.shape[1], a.shape[2]), reverse=True)
        return candidates[0]

    def _postprocess(self, det: np.ndarray, scale_ratio: float, padding, conf_thresh: float, iou_thresh: float, orig_shape):
        preds = det[0].T
        if preds.ndim != 2 or preds.shape[1] < 5:
            print(f"[OpenVINO YOLO] Unexpected output shape: {det.shape}")
            return [], []
        boxes  = preds[:, :4]
        scores = preds[:, 4]
        valid = scores > conf_thresh
        if not np.any(valid):
            return [], []
        boxes  = boxes[valid]
        scores = scores[valid]
        x_c, y_c, w, h = boxes.T
        x1 = x_c - w / 2; y1 = y_c - h / 2
        x2 = x_c + w / 2; y2 = y_c + h / 2
        boxes_xyxy = np.column_stack([x1, y1, x2, y2])
        pad_w, pad_h = padding
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale_ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale_ratio
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])
        idxs = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_thresh, iou_thresh)
        if len(idxs) == 0:
            return [], []
        idxs = np.array(idxs).reshape(-1)
        return boxes_xyxy[idxs].tolist(), scores[idxs].tolist()

    def predict(self, image: np.ndarray, conf: float, iou: float):
        orig_shape = image.shape
        inp, r, pad = self._preprocess_image(image)
        result = self.compiled_model([inp])
        det = self._pick_detection_output(result)
        boxes, scores = self._postprocess(det, r, pad, conf, iou, orig_shape)
        return StandardizedResults(boxes, scores, image)

class DETRDetector:
    def __init__(self, architecture: str, weights: str, device: str, num_labels: int = 1):
        self.architecture = architecture
        self.weights = weights
        self.device = torch.device(device)
        print(f"[DETR] Loading arch='{architecture}', weights='{weights or 'hub'}', num_labels={num_labels}")
        self.processor = AutoImageProcessor.from_pretrained(self.architecture)
        load_from = self.weights if self.weights and Path(self.weights).exists() else self.architecture
        self.model = AutoModelForObjectDetection.from_pretrained(
            load_from,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ).to(self.device).eval()

    def _predict_one(self, image_bgr: np.ndarray, conf: float) -> StandardizedResults:
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image_pil.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf)[0]
        boxes  = results.get("boxes",  torch.empty((0, 4))).detach().cpu().tolist()
        scores = results.get("scores", torch.empty((0,   ))).detach().cpu().tolist()
        return StandardizedResults(boxes, scores, image_bgr)

    def predict(self, image: Union[np.ndarray, List[np.ndarray]], conf: float, iou: float):
        if isinstance(image, list):
            return [self._predict_one(im, conf) for im in image]
        return self._predict_one(image, conf)


class _DefaultPredictorLazy:
    def __init__(self, cfg_path: str, device: str, test_score_thresh: Optional[float] = None, weights_override: Optional[str] = None):
        if not _DETECTRON2_OK:
            raise ImportError("Detectron2 is required for this model.")
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.cfg = LazyConfig.load(cfg_path)

        if test_score_thresh is not None:
            self._try_set_thresh(float(test_score_thresh))

        init_ckpt = None
        if weights_override:
            init_ckpt = weights_override
        else:
            if "train" in self.cfg and hasattr(self.cfg.train, "init_checkpoint"):
                init_ckpt = self.cfg.train.init_checkpoint
            elif "model" in self.cfg and hasattr(self.cfg.model, "weights"):
                init_ckpt = self.cfg.model.weights

        self.model = instantiate(self.cfg.model)
        self.model.to(self.device)
        self.model.eval()
        if init_ckpt:
            DetectionCheckpointer(self.model).load(init_ckpt)

        self.aug = ResizeShortestEdge(short_edge_length=800, max_size=1333)
        self.input_format = "BGR"

    def _try_set_thresh(self, thr: float):
        def _set(obj, path):
            cur = obj
            parts = path.split(".")
            try:
                for p in parts[:-1]:
                    cur = getattr(cur, p)
                setattr(cur, parts[-1], float(thr))
                return True
            except Exception:
                return False
        _set(self.cfg, "model.roi_heads.box_predictor.test_score_thresh") or \
        _set(self.cfg, "model.roi_heads.box_predictor.score_thresh_test") or \
        _set(self.cfg, "model.roi_heads.test_score_thresh") or \
        _set(self.cfg, "model.retinanet.score_threshold")

    def __call__(self, original_image: np.ndarray):
        import torch
        height, width = original_image.shape[:2]
        image = original_image
        if self.input_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        with torch.no_grad():
            preds = self.model([inputs])[0]
        return preds

class VITDetDetector:
    def __init__(self, architecture: str, weights: str, device: str, conf_thresh: float = 0.25):
        if not _DETECTRON2_OK:
            raise ImportError("Detectron2 is required for 'vitdet'. Please install detectron2.")
        if not architecture:
            raise ValueError("VITDetDetector requires 'architecture' (vitdet://... or fs path).")

        from ..utils.vitdet_bootstrap import ensure_vitdet_available, resolve_vitdet_config_path
        proj_root = ensure_vitdet_available()
        if architecture.startswith("vitdet://"):
            architecture = resolve_vitdet_config_path(architecture, proj_root)

        self.device = device if str(device).startswith("cuda") else "cpu"
        self.conf_thresh = float(conf_thresh)

        self.predictor = _DefaultPredictorLazy(
            architecture,
            self.device,
            test_score_thresh=self.conf_thresh,
            weights_override=weights if weights else None
        )

    def _predict_one(self, image_bgr: np.ndarray) -> StandardizedResults:
        outputs = self.predictor(image_bgr)
        inst = outputs["instances"].to("cpu")
        boxes  = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else np.zeros((0, 4), dtype=np.float32)
        scores = inst.scores.numpy() if inst.has("scores") else np.zeros((0,), dtype=np.float32)

        keep = scores >= self.conf_thresh
        boxes = boxes[keep].tolist()
        scores = scores[keep].tolist()
        return StandardizedResults(boxes, scores, image_bgr)

    def predict(self, image: Union[np.ndarray, List[np.ndarray]], conf: float, iou: float):
        self.conf_thresh = min(self.conf_thresh, float(conf))
        if isinstance(image, list):
            return [self._predict_one(im) for im in image]
        return self._predict_one(image)


def _build_predictor_from_yaml(cfg_path: str, device: str, conf: float, weights: Optional[str] = None):
    if not _DETECTRON2_OK:
        raise ImportError("Detectron2 is required for MaskTerial YAML configs.")
    try:
        from ..utils.maskterial_bootstrap import ensure_maskterial_available
        ensure_maskterial_available()
    except Exception as e:
        print("[MaskTerial] WARNING: couldn't import 'maskterial' after bootstrap:", e)
    try:
        from ..utils.m2f_bootstrap import ensure_mask2former_available
        ensure_mask2former_available()
        import types, sys
        try:
            import MultiScaleDeformableAttention as _msda
            shim = types.ModuleType("mask2former.modeling.pixel_decoder.ops.modules")
            shim.MSDeformAttn = getattr(_msda, "MSDeformAttn", None) or getattr(_msda, "MultiScaleDeformableAttention", None)
            if shim.MSDeformAttn is None:
                raise ImportError("MSDeformAttn symbol not found in MultiScaleDeformableAttention")
            sys.modules["mask2former.modeling.pixel_decoder.ops.modules"] = shim
        except Exception as e:
            print("[MaskTerial] WARNING: MSDeformAttn shim not installed:", e)
    except Exception as e:
        print("[MaskTerial] WARNING: couldn't import 'mask2former' after bootstrap:", e)

    cfg = get_cfg()
    try:
        import mask2former
        from mask2former import add_maskformer2_config
        add_maskformer2_config(cfg)
    except Exception as e:
        print("[MaskTerial] WARNING: failed to register Mask2Former config/arch:", e)

    try:
        cfg.set_new_allowed(True)
    except Exception:
        pass

    cfg.merge_from_file(cfg_path)
    try:
        mf = getattr(cfg.MODEL, "MASK_FORMER", None)
        if mf is not None and hasattr(mf, "TEST"):
            if hasattr(mf.TEST, "OBJECT_MASK_THRESHOLD"):
                mf.TEST.OBJECT_MASK_THRESHOLD = float(conf)
            if hasattr(mf.TEST, "DETECTIONS_PER_IMAGE"):
                cur = int(getattr(mf.TEST, "DETECTIONS_PER_IMAGE", 100))
                mf.TEST.DETECTIONS_PER_IMAGE = max(300, cur)
    except Exception:
        pass
    try:
        cfg.set_new_allowed(False)
    except Exception:
        pass

    if weights:
        cfg.MODEL.WEIGHTS = weights

    cfg.MODEL.DEVICE = "cuda" if str(device).startswith("cuda") else "cpu"
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(conf)
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(conf)

    return DefaultPredictor(cfg)





class MaskTerialDetector:
    def __init__(self, architecture: str, weights: str, device: str, conf_thresh: float = 0.25, mask_thresh: float = 0.5):
        if not _DETECTRON2_OK:
            raise ImportError("Detectron2 is required for 'maskterial'. Please install per MaskTerial README.")
        if not architecture:
            raise ValueError("MaskTerialDetector requires 'architecture' (maskterial://... | .py | .yaml).")

        from ..utils.maskterial_bootstrap import ensure_maskterial_available, resolve_maskterial_config_path
        proj_root = ensure_maskterial_available()
        if architecture.startswith("maskterial://"):
            cfg_path = resolve_maskterial_config_path(architecture, proj_root)
        else:
            cfg_path = architecture

        self.device = device if str(device).startswith("cuda") else "cpu"
        self.conf_thresh = float(conf_thresh)
        self.mask_thresh = float(mask_thresh)

        if cfg_path.endswith((".yaml", ".yml")):
            self.predictor = _build_predictor_from_yaml(cfg_path, self.device, self.conf_thresh, weights)
        else:
            self.predictor = _DefaultPredictorLazy(
                cfg_path,
                self.device,
                test_score_thresh=self.conf_thresh,
                weights_override=weights if weights else None
            )

    def _instances_to_xyxy_scores(self, outputs) -> (List[List[float]], List[float]):
        inst = outputs["instances"].to("cpu") if "instances" in outputs else outputs.get("instances", None)
        if inst is None:
            return [], []

        if inst.has("pred_boxes"):
            b = inst.pred_boxes.tensor.numpy()
            s = inst.scores.numpy() if inst.has("scores") else np.ones((b.shape[0],), dtype=np.float32)
            keep = s >= self.conf_thresh
            return b[keep].tolist(), s[keep].tolist()

        if inst.has("pred_masks"):
            ms = inst.pred_masks.numpy()  # (N, H, W) bool/0-1
            s = inst.scores.numpy() if inst.has("scores") else np.ones((ms.shape[0],), dtype=np.float32)
            boxes, scores = [], []
            for m, sc in zip(ms, s):
                if sc < self.conf_thresh:
                    continue
                ys, xs = np.where(m > self.mask_thresh)
                if ys.size == 0 or xs.size == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                boxes.append([x1, y1, x2, y2])
                scores.append(float(sc))
            return boxes, scores

        return [], []

    def _predict_one(self, image_bgr: np.ndarray) -> StandardizedResults:
        outputs = self.predictor(image_bgr)
        boxes, scores = self._instances_to_xyxy_scores(outputs)
        return StandardizedResults(boxes, scores, image_bgr)

    def predict(self, image: Union[np.ndarray, List[np.ndarray]], conf: float, iou: float):
        self.conf_thresh = min(self.conf_thresh, float(conf))
        if isinstance(image, list):
            return [self._predict_one(im) for im in image]
        return self._predict_one(image)

def _resolve_backend_cfg(det_cfg: dict, model_type: str) -> dict:
    backend = det_cfg.get(model_type, {}) or {}
    merged = dict(det_cfg)
    merged.update(backend)
    return merged

def get_detector(config: dict):
    det_cfg    = config['models']['detector']
    model_type = det_cfg.get('type', 'yolo').lower()
    device     = config['device']

    eff = _resolve_backend_cfg(det_cfg, model_type)
    conf_thresh = float(det_cfg.get('conf_thresh', eff.get('conf_thresh', 0.25)))
    iou_thresh  = float(det_cfg.get('iou_thresh',  eff.get('iou_thresh',  0.05)))

    if model_type == 'yolo':
        weights = eff.get('weights') or eff.get('yolo', {}).get('weights')
        if not weights:
            raise ValueError("YOLO detector requires 'models.detector.yolo.weights'.")
        return YOLODetector(weights, device)

    if model_type == 'openvino_yolo':
        weights = eff.get('weights') or eff.get('openvino_yolo', {}).get('weights')
        if not weights:
            raise ValueError("OpenVINO YOLO requires 'models.detector.openvino_yolo.weights'.")
        return FastOpenVINOYOLO(weights, device)

    if model_type in ('detr', 'transformers'):
        architecture = eff.get('architecture') or eff.get('detr', {}).get('architecture')
        if not architecture:
            raise ValueError("'architecture' is required for detector type 'detr'.")
        weights = eff.get('weights') or eff.get('detr', {}).get('weights', "")
        num_labels = int(eff.get('num_labels', det_cfg.get('num_labels', 1)))
        return DETRDetector(architecture, weights, device, num_labels=num_labels)

    if model_type in ('vitdet', 'vit'):
        architecture = eff.get('architecture') or eff.get('vitdet', {}).get('architecture')
        if not architecture:
            raise ValueError("'architecture' is required for detector type 'vitdet'.")
        weights = eff.get('weights') or eff.get('vitdet', {}).get('weights', "")
        return VITDetDetector(architecture, weights, device, conf_thresh=conf_thresh)

    if model_type in ('maskterial', 'maskterial_det', 'maskterial_seg'):
        architecture = eff.get('architecture') or eff.get('maskterial', {}).get('architecture')
        if not architecture:
            raise ValueError("'architecture' is required for detector type 'maskterial'.")
        weights = eff.get('weights') or eff.get('maskterial', {}).get('weights', "")
        mask_thresh = float(eff.get('mask_thresh', eff.get('maskterial', {}).get('mask_thresh', 0.5)))
        return MaskTerialDetector(architecture, weights, device, conf_thresh=conf_thresh, mask_thresh=mask_thresh)

    raise ValueError(f"Unsupported detector type: '{model_type}' in config.")
