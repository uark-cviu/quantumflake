# quantumflake/cli.py

import argparse
import glob
from pathlib import Path

from .pipeline import FlakePipeline
from .utils.io import load_config, merge_configs

# The path to the internal default config
DEFAULT_CONFIG_PATH = Path(__file__).parent / 'cfg' / 'default.yaml'

def main():
    parser = argparse.ArgumentParser(
        description="QuantumFlake: A framework for 2D Material Detection and Classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Predict Command ---
    p_predict = subparsers.add_parser("predict", help="Run inference on images.")
    p_predict.add_argument(
        "source",
        help="Path to an image, directory, or glob pattern (e.g., 'data/*.png')."
    )
    p_predict.add_argument(
        "-c", "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a custom YAML config file. Defaults to the internal default."
    )
    p_predict.add_argument(
        "--opts",
        nargs='*',
        help="Override config options, e.g., device=cuda:0 save_vis=True"
    )

    # --- Placeholder for Train Command ---
    p_train = subparsers.add_parser("train", help="Train a model (detector or classifier).")
    p_train.add_argument("model", choices=['detector', 'classifier'], help="Which model to train.")
    # ... add other training args ...

    args = parser.parse_args()

    # --- Command Logic ---
    if args.command == "predict":
        # Load base config and merge CLI overrides
        config = load_config(args.config)
        if args.opts:
            config = merge_configs(config, args.opts)

        # Find image paths
        image_paths = glob.glob(args.source, recursive=True)
        if not image_paths:
            print(f"Error: No images found at '{args.source}'")
            return

        # Initialize and run pipeline
        pipeline = FlakePipeline(config)
        results = pipeline(image_paths)

        # Print results to console
        print("\n--- Prediction Complete ---")
        for img_path, result_list in zip(image_paths, results):
            print(f"\n[+] Image: {Path(img_path).name}")
            if result_list:
                for i, flake in enumerate(result_list):
                    print(f"  - Flake {i+1}: Class={flake['cls']} (Confidence: {flake['cls_conf']})")
            else:
                print("  No flakes found.")
        print("---------------------------\n")

    elif args.command == "train":
        print(f"Training for '{args.model}' is not yet implemented.")

if __name__ == "__main__":
    main()
