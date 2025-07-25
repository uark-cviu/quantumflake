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
    p_predict.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a custom YAML config file.")
    p_predict.add_argument("--opts", nargs='*', help="Override config options, e.g., device=cuda:0")

    p_train = subparsers.add_parser("train", help="Train a model.")
    train_subparsers = p_train.add_subparsers(dest="model_type", required=True)

    # Train Detector
    p_train_det = train_subparsers.add_parser("detector", help="Train the YOLO detector.")
    p_train_det.add_argument("--data", required=True, help="Path to the dataset YAML file (e.g., coco.yaml).")
    p_train_det.add_argument("--epochs", type=int, default=50)
    p_train_det.add_argument("--imgsz", type=int, default=640)
    p_train_det.add_argument("--device", default="0")

    # Train Classifier
    p_train_cls = train_subparsers.add_parser("classifier", help="Train the ResNet classifier.")
    p_train_cls.add_argument("--data", required=True, help="Path to the root directory of the classified image dataset.")
    p_train_cls.add_argument("--epochs", type=int, default=30)
    p_train_cls.add_argument("--lr", type=float, default=1e-3)
    p_train_cls.add_argument("--batch-size", type=int, default=32)
    p_train_cls.add_argument("--val-split", type=float, default=0.2)
    p_train_cls.add_argument("--device", default="cpu")
    p_train_cls.add_argument("--save-dir", default="runs/classify")
    p_train_cls.add_argument("--num-materials", type=int, default=4, help="Number of materials for the embedding layer.")
    p_train_cls.add_argument("--material-dim", type=int, default=64, help="Dimension of the material embedding vector.")

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)
    elif args.command == "train":
        if args.model_type == "detector":
            run_train_detector(args)
        elif args.model_type == "classifier":
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
                print(f"  - Flake {i+1}: Class={flake['cls']} (Confidence: {flake['cls_conf']})")
        else:
            print("  No flakes found.")

def run_train_detector(args):
    detect.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device)

def run_train_classifier(args):
    classify.train(
        data_dir=args.data, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_split=args.val_split, device=args.device, save_dir=args.save_dir,
        num_materials=args.num_materials, material_dim=args.material_dim
    )

if __name__ == "__main__":
    main()
