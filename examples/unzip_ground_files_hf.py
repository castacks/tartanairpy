import os
import argparse
from os.path import join, relpath, dirname
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# ────────────────────────────────
# Print utilities
# ────────────────────────────────

def print_warn(msg):
    print(f"\033[93m{msg}\033[0m")

def print_error(msg):
    print(f"\033[91m{msg}\033[0m")

def print_highlight(msg):
    print(f"\033[92m{msg}\033[0m")

# ────────────────────────────────
# Unzip preserving relative structure
# ────────────────────────────────

def unzip_to_output(zipfile_path, input_root, output_root, delete_after):
    # Determine relative folder structure
    rel_path = relpath(zipfile_path, input_root)  # e.g. AbandonedCable/Data_diff/P1000/depth_lcam_front.zip
    rel_dir = dirname(rel_path)  # e.g. AbandonedCable/Data_diff/P1000

    dest_dir = join(output_root, rel_dir)
    os.makedirs(dest_dir, exist_ok=True)

    cmd = f'unzip -q -o "{zipfile_path}" -d "{dest_dir}"'
    result = os.system(cmd)

    if result != 0:
        print_error(f"❌ Failed to unzip: {zipfile_path}")
    else:
        print_highlight(f"✅ Unzipped: {zipfile_path} -> {dest_dir}")
        if delete_after:
            os.remove(zipfile_path)
            print_warn(f"🗑️ Deleted: {zipfile_path}")

# ────────────────────────────────
# Discover zip files
# ────────────────────────────────

def collect_zip_files(input_dir):
    zipfiles = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zip'):
                zipfiles.append(join(root, file))
    return zipfiles

# ────────────────────────────────
# Main function
# ────────────────────────────────

def main(args):
    print_warn("⚠️  Unzipping will overwrite existing files in output directory...")

    zipfiles = collect_zip_files(args.input_dir)
    print(f"📦 Found {len(zipfiles)} zip files.")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                unzip_to_output, zf, args.input_dir, args.output_dir, args.delete_zip
            ): zf for zf in zipfiles
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Unzipping"):
            pass

    print_highlight("🎉 All unzips completed.")

# ────────────────────────────────
# Entry point
# ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded unzip for HuggingFace dataset, preserving folder structure in a separate output dir")
    parser.add_argument("--input_dir", required=True, help="Directory containing recursively downloaded .zip files from HuggingFace")
    parser.add_argument("--output_dir", required=True, help="Directory where extracted content should be placed")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--delete_zip", action="store_true", help="Delete original zip files after extracting")
    args = parser.parse_args()

    main(args)
