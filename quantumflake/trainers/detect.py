from pathlib import Path
import yaml
from ultralytics import YOLO
from math import ceil

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
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg, LazyConfig, instantiate
    from detectron2.data import DatasetCatalog
    from detectron2 import model_zoo
    from detectron2.engine import AMPTrainer, DefaultTrainer, SimpleTrainer, create_ddp_model, default_writers, hooks
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
    from detectron2.data.datasets import register_coco_instances
except Exception:
    _DETECTRON2_OK = False


def _register_coco_once(name, ann_json, images_dir):
    if name not in DatasetCatalog.list():
        register_coco_instances(name, {}, ann_json, images_dir)


def _resolve_vitdet_architecture(architecture):
    if architecture.startswith("vitdet://"):
        from ..utils.vitdet_bootstrap import ensure_vitdet_available, resolve_vitdet_config_path

        proj_root = ensure_vitdet_available()
        return resolve_vitdet_config_path(architecture, proj_root)
    if architecture.startswith("model_zoo:"):
        return model_zoo.get_config_file(architecture.split("model_zoo:", 1)[1])
    return architecture


def _count_coco_images(ann_json):
    with open(ann_json, "r") as f:
        return len(json.load(f)["images"])


def _coco_has_segmentation(ann_json):
    with open(ann_json, "r") as f:
        anns = json.load(f).get("annotations", [])
    return any(ann.get("segmentation") for ann in anns)


def _train_vitdet_lazy(
    cfg_path,
    weights,
    train_name,
    val_name,
    train_json,
    device,
    epochs,
    base_lr,
    ims_per_batch,
    eval_period,
    out_dir,
):
    cfg = LazyConfig.load(cfg_path)
    cfg.dataloader.train.dataset.names = train_name
    cfg.dataloader.test.dataset.names = val_name
    cfg.dataloader.train.total_batch_size = ims_per_batch
    cfg.dataloader.train.num_workers = 0
    cfg.dataloader.test.num_workers = 0
    cfg.train.device = "cuda" if str(device).lower() != "cpu" else "cpu"
    if cfg.train.device == "cpu":
        cfg.train.amp.enabled = False
    cfg.train.output_dir = out_dir
    cfg.train.eval_period = eval_period
    cfg.train.log_period = 1
    cfg.train.max_iter = max(1, epochs * ceil(_count_coco_images(train_json) / max(ims_per_batch, 1)))
    if hasattr(cfg.model, "roi_heads"):
        cfg.model.roi_heads.num_classes = 1
        if not _coco_has_segmentation(train_json):
            cfg.model.roi_heads.mask_in_features = None
            cfg.model.roi_heads.mask_pooler = None
            cfg.model.roi_heads.mask_head = None
            cfg.dataloader.train.mapper.use_instance_mask = False
            cfg.dataloader.train.mapper.recompute_boxes = False
    if weights:
        cfg.train.init_checkpoint = weights
    cfg.optimizer.lr = base_lr

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)
    model = create_ddp_model(model, **cfg.train.ddp)

    trainer_cls = AMPTrainer if cfg.train.amp.enabled else SimpleTrainer
    trainer = trainer_cls(model, train_loader, optimizer)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.train.output_dir, trainer=trainer)

    def _do_test():
        test_loader = instantiate(cfg.dataloader.test)
        evaluator = instantiate(cfg.dataloader.evaluator)
        ret = inference_on_dataset(model, test_loader, evaluator)
        print_csv_format(ret)
        return ret

    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.PeriodicCheckpointer(
            checkpointer,
            period=int(cfg.train.checkpointer.period),
            max_to_keep=int(cfg.train.checkpointer.max_to_keep),
        ),
        hooks.EvalHook(cfg.train.eval_period, _do_test) if cfg.train.eval_period > 0 else None,
        hooks.PeriodicWriter(default_writers(cfg.train.output_dir, cfg.train.max_iter), period=cfg.train.log_period),
    ])

    if cfg.train.init_checkpoint:
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
    trainer.train(0, cfg.train.max_iter)
    checkpointer.save("model_final")

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

    _register_coco_once(train_name, train_json, train_images)
    _register_coco_once(val_name, val_json, val_images)

    cfg_path = _resolve_vitdet_architecture(architecture)

    print("\n--- Starting ViTDet Training (Detectron2) ---")
    if cfg_path.endswith(".py"):
        _train_vitdet_lazy(
            cfg_path=cfg_path,
            weights=weights,
            train_name=train_name,
            val_name=val_name,
            train_json=train_json,
            device=device,
            epochs=epochs,
            base_lr=base_lr,
            ims_per_batch=ims_per_batch,
            eval_period=eval_period,
            out_dir=out_dir,
        )
        print("\n--- ViTDet Training Finished ---")
        print(f"Artifacts saved to: {out_dir}")
        return

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    elif architecture.startswith("model_zoo:"):
        zoo_key = architecture.split("model_zoo:", 1)[1]
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_key)
    else:
        raise ValueError("ViTDet training with local YAML configs requires --vit-weights")
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST  = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    n_imgs = _count_coco_images(train_json)
    iters_per_epoch = max(1, ceil(n_imgs / max(ims_per_batch, 1)))
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

    trainer = _Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("\n--- ViTDet Training Finished ---")
    print(f"Artifacts saved to: {out_dir}")
