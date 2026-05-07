import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import re
from tqdm import tqdm
import multiprocessing
import torch
import torchio as tio


def load_metadata_map(csv_path):
    print(f"[Main] Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Identify and drop rows missing critical spacing data (Matches create_final_shards_3.py)
    missing_mask = df[['XYSpacing', 'ZSpacing']].isna().any(axis=1)
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"[Main] ⚠️ Skipping {missing_count} volumes due to missing XYSpacing or ZSpacing.")
        df = df[~missing_mask]

    # Extract base scan group ID using regex to remove trailing _number
    # e.g., 'valid_1_a_1.nii.gz' -> 'valid_1_a'
    df['ScanGroup'] = df['VolumeName'].apply(
        lambda x: re.sub(r'_\d+$', '', x.replace('.nii.gz', ''))
    )

    # Sort by ScanGroup, then ZSpacing (ascending), then VolumeName
    # Sorting ZSpacing ascending brings the smallest value (highest resolution) to the top.
    # Sorting by VolumeName ensures deterministic tie-breaking if resolutions are identical.
    df = df.sort_values(by=['ScanGroup', 'ZSpacing', 'VolumeName'], ascending=[True, True, True])

    # Keep only the first entry for each ScanGroup
    df_highest_res = df.drop_duplicates(subset=['ScanGroup'], keep='first')

    print(f"[Main] Filtered to highest resolution per scan group. Kept {len(df_highest_res)} volumes out of {len(df)}.")

    # Include ZSpacing and XYSpacing in the dictionary
    meta_map = df_highest_res.set_index('VolumeName')[
        ['RescaleSlope', 'RescaleIntercept', 'ZSpacing', 'XYSpacing']].to_dict('index')
    print(f"[Main] Metadata map populated for {len(meta_map)} highest-res volumes.")
    return meta_map


def load_blocklist(txt_path):
    """Loads the list of brain scans/volumes to ignore."""
    blocklist = set()
    if txt_path and os.path.exists(txt_path):
        print(f"[Main] Loading blocklist from {txt_path}...")
        with open(txt_path, 'r') as f:
            for line in f:
                # 1. Strip whitespace
                clean_line = line.strip()
                # 2. Get just the filename
                filename = os.path.basename(clean_line)
                # 3. Remove extension to get the raw ID
                vol_id = filename.replace('.nii.gz', '')
                blocklist.add(vol_id)

        print(f"[Main] Blocklist contains {len(blocklist)} volumes.")
    return blocklist


def process_volume(args):
    """
    Worker function to process a single NIfTI file, apply metadata transforms,
    resample to 1mm isotropic, and save as .npy.
    """
    nifti_path, metadata, output_dir = args
    filename = os.path.basename(nifti_path)

    # Skip if output already exists (resume capability)
    out_name = filename.replace('.nii.gz', '.npy')
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        return

    if filename not in metadata:
        print(f"[Worker] ERROR: Missing metadata for {filename}")
        return

    try:
        # 1. Load Data
        img = nib.load(nifti_path)
        raw_data = img.get_fdata(dtype=np.float32)

        # 2. Metadata Transform (HU Conversion)
        params = metadata[filename]
        slope = float(params['RescaleSlope'])
        inter = float(params['RescaleIntercept'])
        hu_data = ((raw_data * slope) + inter).astype(np.float32)

        # 3. Parse Spacing and Construct Trusted Affine
        original_z_spacing = float(params['ZSpacing'])
        xy_raw = params['XYSpacing']
        if isinstance(xy_raw, str):
            xy_clean = xy_raw.strip('[]').replace("'", "").split(',')
            x_spacing = float(xy_clean[0])
            y_spacing = float(xy_clean[1]) if len(xy_clean) > 1 else x_spacing
        else:
            x_spacing = y_spacing = float(xy_raw)

        trusted_affine = np.diag([x_spacing, y_spacing, original_z_spacing, 1.0])

        # 4. Build TorchIO Subject & Apply 1mm Isotropic Resampling
        subject = tio.Subject(
            ct=tio.ScalarImage(tensor=torch.from_numpy(hu_data).unsqueeze(0), affine=trusted_affine)
        )

        resampler = tio.Resample(1.0, image_interpolation='bspline')
        resampled = resampler(subject)

        # Extract resampled numpy array and remove channel dim -> Shape: (H, W, Depth)
        resampled_hu = resampled['ct'].data.squeeze(0).numpy()

        # 5. Optimize & Format
        # Shape is (H, W, Depth) -> Transpose to (Depth, H, W) for PyTorch
        final_data = resampled_hu.transpose(2, 0, 1).astype(np.float16)

        # 6. Save as NPY
        np.save(out_path, final_data)

    except Exception as e:
        print(f"[Worker] Error processing {filename}: {e}")


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

    # Filter out files that don't have valid spacing metadata OR aren't the highest resolution
    valid_files = [f for f in valid_files if os.path.basename(f) in meta_map]
    print(f"[Main] {len(valid_files)} files remain to be processed.")

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