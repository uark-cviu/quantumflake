import argparse
import glob
from pathlib import Path
import os

from .pipeline import FlakePipeline
from .utils.io import load_config, merge_configs
from .trainers import detect, classify

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'cfg' / 'default.yaml'

def main():
    parser = argparse.ArgumentParser(
        description="QuantumFlake: A framework for 2D Material Detection and Classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_predict = subparsers.add_parser("predict", help="Run inference on images.")
    p_predict.add_argument("source", help="Path to an image, directory, or glob pattern.")
    p_predict.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a YAML config file.")
    p_predict.add_argument("--opts", nargs='*', help="Override config options, e.g., models.detector.type=detr device=cuda:0")
    p_train = subparsers.add_parser("train", help="Train a model.")
    train_subparsers = p_train.add_subparsers(dest="train_what", required=True)
    p_train_det = train_subparsers.add_parser("detector", help="Train a detector backend (yolo | detr | vitdet).")
    p_train_det.add_argument("--type", required=True, choices=["yolo", "detr", "vitdet"],
                             help="Detector backend to train.")
    p_train_det.add_argument("--device", default="0",
                             help="YOLO/ViTDet: GPU index(s) '0' or '0,1' or 'cpu'; DETR: 'cuda:0' or 'cpu'.")
    p_train_det.add_argument("--epochs", type=int, default=50)
    p_train_det.add_argument("--output-dir", default="runs/detect_train")
    p_train_det.add_argument("--data", help="YOLO dataset YAML (for YOLO backend).")
    p_train_det.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    p_train_det.add_argument("--weights", default="yolo11n.pt", help="YOLO init weights.")
    p_train_det.add_argument("--architecture", help="HF model id (e.g., facebook/detr-resnet-50) for DETR.")
    p_train_det.add_argument("--train-json", help="COCO train annotations .json")
    p_train_det.add_argument("--train-images", help="COCO train images dir")
    p_train_det.add_argument("--val-json", help="COCO val annotations .json")
    p_train_det.add_argument("--val-images", help="COCO val images dir")
    p_train_det.add_argument("--num-labels", type=int, default=1, help="Number of foreground classes.")
    p_train_det.add_argument("--lr", type=float, default=2e-4)
    p_train_det.add_argument("--batch-size", type=int, default=8)
    p_train_det.add_argument("--vit-arch", help="Detectron2 config: 'model_zoo:...yaml' or local .yaml")
    p_train_det.add_argument("--vit-weights", default="", help="Detectron2 .pth checkpoint to start from (optional).")
    p_train_det.add_argument("--train-name", default="flakes_train", help="Detectron2 train dataset name.")
    p_train_det.add_argument("--val-name", default="flakes_val", help="Detectron2 val dataset name.")
    p_train_det.add_argument("--base-lr", type=float, default=0.0002)
    p_train_det.add_argument("--ims-per-batch", type=int, default=8)
    p_train_det.add_argument("--eval-period", type=int, default=2000)
    p_train_cls = train_subparsers.add_parser("classifier", help="Train the ResNet classifier.")
    p_train_cls.add_argument("--data", required=True, help="Path to the root directory of the classified image dataset.")
    p_train_cls.add_argument("--epochs", type=int, default=30)
    p_train_cls.add_argument("--lr", type=float, default=1e-3)
    p_train_cls.add_argument("--batch-size", type=int, default=32)
    p_train_cls.add_argument("--val-split", type=float, default=0.2)
    p_train_cls.add_argument("--device", default="cpu")
    p_train_cls.add_argument("--save-dir", default="runs/classify")
    p_train_cls.add_argument("--num-materials", type=int, default=2, help="Number of materials for the embedding layer.")
    p_train_cls.add_argument("--material-dim", type=int, default=64, help="Dimension of the material embedding vector.")
    p_train_cls.add_argument("--freeze-cnn", action="store_true", help="Freeze CNN backbone during training.")
    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)
    elif args.command == "train":
        if args.train_what == "detector":
            run_train_detector(args)
        elif args.train_what == "classifier":
            run_train_classifier(args)

def run_predict(args):
    config = load_config(args.config)
    if args.opts:
        config = merge_configs(config, args.opts)

    source_path = Path(args.source)
    image_paths = []
    if source_path.is_dir():
        supported_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in supported_exts:
            image_paths.extend(glob.glob(os.path.join(args.source, '**', ext), recursive=True))
    else:
        image_paths = glob.glob(args.source, recursive=True)
    if not image_paths:
        print(f"Error: No images found at '{args.source}'")
        return

    pipeline = FlakePipeline(config)
    results = pipeline(image_paths)
    print("\n--- Prediction Complete ---")
    for img_path, result_list in zip(image_paths, results):
        print(f"\n[+] Image: {Path(img_path).name}")
        if result_list:
            for i, flake in enumerate(result_list):
                print(f"  - Flake {i+1}: Class={flake['cls']} (Conf: {flake['cls_conf']}) DetConf={flake['det_conf']}")
        else:
            print("  No flakes found.")

def run_train_detector(args):
    if args.type == "yolo":
        if not args.data:
            raise SystemExit("--data is required for YOLO training")
        detect.train_yolo(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            device=args.device, weights=args.weights, project=args.output_dir
        )
    elif args.type == "detr":
        required = [args.architecture, args.train_json, args.train_images, args.val_json, args.val_images]
        if any(x is None for x in required):
            raise SystemExit("--architecture, --train-json, --train-images, --val-json, --val-images are required for DETR training")
        detect.train_detr(
            architecture=args.architecture,
            train_json=args.train_json, train_images=args.train_images,
            val_json=args.val_json, val_images=args.val_images,
            num_labels=args.num_labels, device=args.device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            out_dir=args.output_dir
        )
    elif args.type == "vitdet":
        required = [args.vit_arch, args.train_json, args.train_images, args.val_json, args.val_images]
        if any(x is None for x in required):
            raise SystemExit("--vit-arch, --train-json, --train-images, --val-json, --val-images are required for ViTDet training")
        detect.train_vitdet(
            architecture=args.vit_arch, weights=args.vit_weights,
            train_name=args.train_name, val_name=args.val_name,
            train_json=args.train_json, train_images=args.train_images,
            val_json=args.val_json, val_images=args.val_images,
            device=args.device, epochs=args.epochs,
            base_lr=args.base_lr, ims_per_batch=args.ims_per_batch,
            eval_period=args.eval_period, out_dir=args.output_dir
        )

def run_train_classifier(args):
    classify.train(
        data_dir=args.data, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_split=args.val_split, device=args.device, save_dir=args.save_dir,
        num_materials=args.num_materials, material_dim=args.material_dim,
        freeze_cnn=args.freeze_cnn
    )

if __name__ == "__main__":
    main()
