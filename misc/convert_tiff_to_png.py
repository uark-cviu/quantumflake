# tools/convert_tiff_to_png.py

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings

# Pillow can sometimes raise DecompressionBombWarning for large images,
# we can disable it for this trusted, specific use case.
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

def convert_tiff_to_png(input_dir: str, output_dir: str):
    """
    Finds all .tiff and .tif files in a directory and its subdirectories,
    extracts every frame from each file, and saves them as individual .png files.

    Args:
        input_dir (str): The path to the directory containing .tiff files.
        output_dir (str): The path to the directory where .png files will be saved.
    """
    # --- Setup Paths ---
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.is_dir():
        print(f"Error: Input directory not found at '{in_path}'")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted PNG files to: {out_path}")

    # --- Find all TIFF files ---
    tiff_files = list(in_path.rglob("*.tif")) + list(in_path.rglob("*.tiff"))
    
    if not tiff_files:
        print(f"No .tiff or .tif files found in '{in_path}'.")
        return

    print(f"Found {len(tiff_files)} TIFF files to process.")

    # --- Conversion Loop ---
    total_images_saved = 0
    for tiff_path in tqdm(tiff_files, desc="Processing TIFF files"):
        try:
            with Image.open(tiff_path) as img:
                # img.n_frames tells us how many images are in the stack
                for i in range(img.n_frames):
                    # Select the i-th frame in the TIFF stack
                    img.seek(i)

                    # Define a unique output path for each frame
                    # e.g., 'my_image.tif' -> 'my_image_p0.png', 'my_image_p1.png'
                    output_filename = f"{tiff_path.stem}_p{i}.png"
                    final_output_path = out_path / output_filename

                    # Convert the frame to RGB and save as PNG
                    img.convert("RGB").save(final_output_path, "PNG")
                    total_images_saved += 1

        except Exception as e:
            print(f"\nCould not convert file '{tiff_path.name}'. Reason: {e}")
    
    print(f"\nConversion complete. Saved a total of {total_images_saved} PNG images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A standalone script to convert all frames from TIFF images in a directory to PNG format."
    )
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Path to the directory containing your multi-page .tiff files."
    )
    parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Path to the directory where the converted .png files will be saved."
    )
    args = parser.parse_args()

    convert_tiff_to_png(args.input_dir, args.output_dir)
