import os
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import multiprocessing
import torch
import torchio as tio

def load_metadata_map(csv_path):
    print(f"[Main] Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Identify and drop rows missing the final spacing data
    missing_mask = df['final_spacing'].isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"[Main] ⚠️ Skipping {missing_count} volumes due to missing final_spacing.")
        df = df[~missing_mask]

    # Use VolumeAcc_DEID as the key (matches the .npz filenames exactly, e.g., 'trn19675.npz')
    meta_map = df.set_index('VolumeAcc_DEID')[['final_spacing']].to_dict('index')
    print(f"[Main] Metadata loaded for {len(meta_map)} volumes.")
    return meta_map


def load_blocklist(txt_path):
    """Loads the list of brain scans/volumes to ignore."""
    blocklist = set()
    if txt_path and os.path.exists(txt_path):
        print(f"[Main] Loading blocklist from {txt_path}...")
        with open(txt_path, 'r') as f:
            for line in f:
                clean_line = line.strip()
                filename = os.path.basename(clean_line)
                vol_id = filename.replace('.npz', '')
                blocklist.add(vol_id)
        print(f"[Main] Blocklist contains {len(blocklist)} volumes.")
    return blocklist


def process_volume(args):
    """
    Worker function to process a single NPZ file, apply 1mm isotropic resampling,
    and save as .npy.
    """
    file_path, metadata, output_dir = args
    filename = os.path.basename(file_path)

    # Skip if output already exists (resume capability)
    out_name = filename.replace('.npz', '.npy')
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        return

    if filename not in metadata:
        print(f"[Worker] ERROR: Missing metadata for {filename}")
        return

    try:
        # 1. Load Data
        npz_data = np.load(file_path)
        # Grab the first array in the .npz archive
        raw_data = npz_data[npz_data.files[0]].astype(np.float32)

        # 2. Construct Trusted Affine using the provided 0.8mm isotropic spacing
        params = metadata[filename]
        spacing = float(params['final_spacing'])
        trusted_affine = np.diag([spacing, spacing, spacing, 1.0])

        # 3. Build TorchIO Subject & Apply 1mm Isotropic Resampling
        subject = tio.Subject(
            ct=tio.ScalarImage(tensor=torch.from_numpy(raw_data).unsqueeze(0), affine=trusted_affine)
        )

        resampler = tio.Resample(1.0, image_interpolation='bspline')
        resampled = resampler(subject)

        # Extract resampled numpy array and remove channel dim
        resampled_hu = resampled['ct'].data.squeeze(0).numpy()

        # 4. Optimize & Format
        # TorchIO outputs (X, Y, Z). Transpose to (Depth, H, W) for PyTorch
        final_data = resampled_hu.transpose(2, 0, 1).astype(np.float16)

        # 5. Save as NPY
        np.save(out_path, final_data)

    except Exception as e:
        print(f"[Worker] Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blocklist", help="Path to text file containing volumes to ignore", default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Meta & Blocklist
    meta_map = load_metadata_map(args.metadata)
    blocklist = load_blocklist(args.blocklist)

    # 2. Find Files
    files = glob.glob(os.path.join(args.data_dir, '**', '*.npz'), recursive=True)
    print(f"[Main] Found {len(files)} total files.")

    # 3. Filter Blocklist
    valid_files = []
    if blocklist:
        print("[Main] Filtering blocklist...")
        for f in files:
            name = os.path.basename(f)
            vol_id = name.replace('.npz', '')

            if vol_id in blocklist or name in blocklist:
                continue
            valid_files.append(f)
        print(f"[Main] {len(valid_files)} files remain after filtering.")
    else:
        valid_files = files

    # Filter out files that don't have valid spacing metadata
    valid_files = [f for f in valid_files if os.path.basename(f) in meta_map]
    print(f"[Main] {len(valid_files)} files remain after removing missing metadata.")

    # 4. Prepare Args for Workers
    task_args = []
    for f in valid_files:
        task_args.append((f, meta_map, args.output_dir))

    print("[Main] Starting processing...")

    # Run Parallel
    with multiprocessing.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_volume, task_args), total=len(valid_files)))

    print("[Main] Validation preprocessing complete!")

if __name__ == "__main__":
    main()