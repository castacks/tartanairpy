import os
import re
import argparse
from os.path import join, isfile, basename
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
# Unzip single file
# ────────────────────────────────

def unzip_file(zipfile_path, data_root):
    filename = basename(zipfile_path)

    # Match Case 1: Trajectory zips
    match_traj = re.match(r"(.+?)_Data_(\w+)_((?:P\d+))_.+\.zip", filename)

    # Match Case 2: Label zips
    match_label = re.match(r"(.+?)_(seg_labels|seg_label_map)\.zip", filename)

    # Match Case 3: PCD zips
    match_pcd = re.match(r"(.+?)_\1_(rgb|sem)_pcd\.zip", filename)

    if match_traj:
        env_name, data_type, traj_name = match_traj.groups()
        dest_dir = join(data_root, env_name, f"Data_{data_type}", traj_name)

    elif match_label:
        env_name, _ = match_label.groups()
        dest_dir = join(data_root, env_name)

    elif match_pcd:
        env_name, _ = match_pcd.groups()
        dest_dir = join(data_root, env_name)

    else:
        print_error(f"❌ Could not parse zip file name: {filename}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    cmd = f'unzip -q -o "{zipfile_path}" -d "{dest_dir}"'
    result = os.system(cmd)

    if result != 0:
        print_error(f"❌ Failed to unzip: {filename}")
    else:
        print_highlight(f"✅ Unzipped: {filename} -> {dest_dir}")

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
    print_warn("⚠️  Unzipping will overwrite existing files...")

    zipfiles = collect_zip_files(args.input_dir)
    print(f"📦 Found {len(zipfiles)} zip files.")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(unzip_file, zf, args.output_dir): zf for zf in zipfiles}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Unzipping"):
            pass

    print_highlight("🎉 All unzips completed.")

# ────────────────────────────────
# Entry point
# ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded unzipper for structured TartanAir zips")
    parser.add_argument("--input_dir", required=True, help="Directory containing .zip files")
    parser.add_argument("--output_dir", required=True, help="Root directory to extract into")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    args = parser.parse_args()

    main(args)
