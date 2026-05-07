import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
from tqdm import tqdm
import multiprocessing


def load_metadata_map(csv_path):
    # Same as your training script
    print(f"[Main] Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df.set_index('VolumeName')[['RescaleSlope', 'RescaleIntercept']].to_dict('index')


def load_blocklist(txt_path):
    """Loads the list of brain scans/volumes to ignore."""
    blocklist = set()
    if txt_path and os.path.exists(txt_path):
        print(f"[Main] Loading blocklist from {txt_path}...")
        with open(txt_path, 'r') as f:
            for line in f:
                # 1. Strip whitespace
                clean_line = line.strip()
                # 2. Get just the filename (removes "train/train_10100/...")
                filename = os.path.basename(clean_line)
                # 3. Remove extension to get the raw ID
                vol_id = filename.replace('.nii.gz', '')
                blocklist.add(vol_id)

        print(f"[Main] Blocklist contains {len(blocklist)} volumes.")
    return blocklist


def process_volume(args):
    """
    Worker function to process a single NIfTI file and save as .npy
    """
    nifti_path, metadata, output_dir = args
    filename = os.path.basename(nifti_path)

    # Skip if output already exists (resume capability)
    out_name = filename.replace('.nii.gz', '.npy')
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        return

    try:
        # 1. Load Data
        img = nib.load(nifti_path)
        raw_data = img.get_fdata(dtype=np.float32)

        # 2. Metadata Transform
        if filename in metadata:
            slope = float(metadata[filename]['RescaleSlope'])
            inter = float(metadata[filename]['RescaleIntercept'])
            hu_data = (raw_data * slope) + inter
        else:
            # Fallback if metadata missing (or log error)
            hu_data = raw_data
            print("ERROR: Missing metadata")
            print(raw_data)

        # 3. Optimize (Float16)
        # Shape is (H, W, Depth) -> Transpose to (Depth, H, W) for PyTorch
        hu_data = hu_data.transpose(2, 0, 1).astype(np.float16)

        # 4. Save
        np.save(out_path, hu_data)

    except Exception as e:
        print(f"Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blocklist", help="Path to text file containing volumes to ignore")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Meta & Blocklist
    meta_map = load_metadata_map(args.metadata)
    blocklist = load_blocklist(args.blocklist)

    # 2. Find Files
    files = glob.glob(os.path.join(args.data_dir, '**', '*.nii.gz'), recursive=True)
    print(f"[Main] Found {len(files)} total files.")

    # 3. Filter Blocklist
    valid_files = []
    if blocklist:
        print("[Main] Filtering blocklist...")
        for f in files:
            name = os.path.basename(f)
            vol_id = name.replace('.nii.gz', '')

            # Check against blocklist (both full name and ID)
            if vol_id in blocklist or name in blocklist:
                continue
            valid_files.append(f)
        print(f"[Main] {len(valid_files)} files remain after filtering.")
    else:
        valid_files = files

    # 4. Prepare Args for Workers
    task_args = []
    for f in valid_files:
        # Just pass the map, worker handles lookup
        task_args.append((f, meta_map, args.output_dir))

    print("Starting processing...")

    # Run Parallel
    with multiprocessing.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_volume, task_args), total=len(valid_files)))


if __name__ == "__main__":
    main()