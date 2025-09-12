from pathlib import Path
import yaml
from ultralytics import YOLO

def train_yolo(data, epochs=50, imgsz=640, device="0", weights="yolo11n.pt", project="runs/detect_train"):
    print("\n--- Starting YOLO Detector Training ---")
    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found at: {data}")
    with open(data_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        num_classes = len(data_yaml.get('names', []))
        if num_classes != 1:
            print(f"WARNING: Expected 1 class for flake detection, but found {num_classes} in {data}. Proceeding anyway.")
    model = YOLO(weights)
    print(f"Starting training on device '{device}' with dataset '{data}'...")
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=project,
        name=data_path.stem
    )
    print("\n--- YOLO Detector Training Finished ---")
    print(f"Results and best model saved in '{project}/{data_path.stem}'")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import json
import os

class CocoDetectionHF(Dataset):
    def __init__(self, images_dir, ann_json, processor):
        self.images_dir = images_dir
        self.processor = processor
        with open(ann_json, "r") as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}
        self.ann_by_img = {}
        for ann in coco["annotations"]:
            self.ann_by_img.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.ann_by_img.get(img_id, [])
        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "iscrowd": a.get("iscrowd", 0),
                    "category_id": a["category_id"],
                    "bbox": a["bbox"],  # [x, y, w, h] in pixels
                    "area": a.get("area", a["bbox"][2]*a["bbox"][3]),
                } for a in anns if a.get("bbox") is not None
            ]
        }
        return image, target

def _collate_fn_hf(batch, processor, device):
    images, targets = list(zip(*batch))
    enc = processor(images=list(images), annotations=list(targets), return_tensors="pt")
    enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
    return enc

def train_detr(
    architecture,
    train_json, train_images,
    val_json, val_images,
    num_labels=1, device="cuda:0",
    epochs=50, lr=2e-4, batch_size=8,
    out_dir="runs/detect_train/detr",
):
    print("\n--- Starting DETR Training (HF) ---")
    device = torch.device(device)

    processor = AutoImageProcessor.from_pretrained(architecture)
    model = AutoModelForObjectDetection.from_pretrained(
        architecture, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

    train_ds = CocoDetectionHF(train_images, train_json, processor)
    val_ds   = CocoDetectionHF(val_images,   val_json,   processor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: _collate_fn_hf(b, processor, device))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: _collate_fn_hf(b, processor, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 50 == 0:
                print(f"[Epoch {epoch}] step {step} | train loss: {running/step:.4f}")
        model.eval()
        val_loss = 0.0; vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                vsteps += 1
        val_loss = val_loss / max(vsteps, 1)
        print(f"[Epoch {epoch}] val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_path = out_dir / "best_detr"
            save_path.mkdir(exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  -> New best saved to: {save_path}")

    print("\n--- DETR Training Finished ---")
    print(f"Best val loss: {best_val:.4f}")

_DETECTRON2_OK = True
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator
    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
except Exception:
    _DETECTRON2_OK = False

def train_vitdet(
    architecture, weights,
    train_name, val_name,
    train_json, train_images,
    val_json, val_images,
    device="0", epochs=50,
    base_lr=2e-4, ims_per_batch=8,
    eval_period=2000, out_dir="runs/detect_train/vitdet",
):
    if not _DETECTRON2_OK:
        raise ImportError("Detectron2 is required for ViTDet training. Please install detectron2.")

    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name,   {}, val_json,   val_images)

    cfg = get_cfg()
    if architecture.startswith("model_zoo:"):
        zoo_key = architecture.split("model_zoo:", 1)[1]
        cfg.merge_from_file(model_zoo.get_config_file(zoo_key))
        if not weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_key)
    else:
        cfg.merge_from_file(architecture)
        if not weights:
            raise ValueError("ViTDet training with local config requires --vit-weights")
    if weights:
        cfg.MODEL.WEIGHTS = weights

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST  = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    import json as _json, os as _os
    with open(train_json, "r") as f:
        n_imgs = len(_json.load(f)["images"])
    iters_per_epoch = max(1, n_imgs // ims_per_batch)
    cfg.SOLVER.MAX_ITER = epochs * iters_per_epoch
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(1000, iters_per_epoch // 5)

    cfg.MODEL.DEVICE = "cuda" if str(device).lower() != "cpu" else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.OUTPUT_DIR = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    class _Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    print("\n--- Starting ViTDet Training (Detectron2) ---")
    trainer = _Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("\n--- ViTDet Training Finished ---")
    print(f"Artifacts saved to: {out_dir}")
