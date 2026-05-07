import argparse
import os
import glob
import json
import multiprocessing
import numpy as np
import nibabel as nib
from collections import defaultdict

# --- Constants ---
# TotalSegmentator uses 1-117 (0 is background). We will track 0-117 for completeness.
NUM_TS_CLASSES = 118
# Hardcoded ReX Classes based on your dataloader
REX_CLASSES = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"
]


def load_rex_metadata(json_path, seg_dir):
    """Reused from create_final_shards_2.py to map ReX files to categories"""
    rex_lookup = {}
    if not json_path or not os.path.exists(json_path):
        return rex_lookup

    with open(json_path, 'r') as f:
        data = json.load(f)

    all_items = []
    if isinstance(data, list):
        all_items = data
    elif isinstance(data, dict):
        if data:
            first_val = next(iter(data.values()))
            if isinstance(first_val, list):
                for lst in data.values():
                    if isinstance(lst, list): all_items.extend(lst)
            else:
                all_items = list(data.values())

    for entry in all_items:
        if isinstance(entry, dict) and entry.get('name'):
            rex_lookup[entry['name']] = {
                'categories': entry.get('categories', {}),
                'path': None
            }

    if seg_dir and os.path.exists(seg_dir):
        seg_files = glob.glob(os.path.join(seg_dir, "**", "*.nii.gz"), recursive=True)
        for f in seg_files:
            name = os.path.basename(f)
            if name in rex_lookup:
                rex_lookup[name]['path'] = f
    return rex_lookup


def scan_file_directory(root_dir):
    """Reused from create_final_shards_2.py to find TS masks"""
    lookup = {}
    if root_dir and os.path.exists(root_dir):
        files = glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
        for f in files:
            lookup[os.path.basename(f)] = f
    return lookup


def worker_routine(worker_id, ts_files, rex_files, rex_lookup):
    """
    Processes a chunk of TS and ReX NIfTI files to compute global voxel counts and slice presence.
    """
    # Trackers for this specific worker
    worker_stats = {
        'ts_voxel_counts': np.zeros(NUM_TS_CLASSES, dtype=np.int64),
        'ts_slice_counts': np.zeros(NUM_TS_CLASSES, dtype=np.int64),
        'ts_total_voxels': 0,

        'rex_voxel_counts': {c: 0 for c in REX_CLASSES},
        'rex_slice_counts': {c: 0 for c in REX_CLASSES},
        'rex_total_voxels': 0
    }

    # --- 1. Process TotalSegmentator Masks ---
    for ts_path in ts_files:
        try:
            img = nib.load(ts_path)
            # Load as uint8 since TS classes max out at 117
            vol = img.get_fdata().astype(np.uint8)

            # Global voxel count for this volume
            worker_stats['ts_total_voxels'] += vol.size

            # Count voxels per class using bincount for extreme speed
            counts = np.bincount(vol.flatten(), minlength=NUM_TS_CLASSES)
            worker_stats['ts_voxel_counts'] += counts[:NUM_TS_CLASSES]

            # Count slice presence: for every slice z, check which classes exist
            depth = vol.shape[2]
            for z in range(depth):
                slice_data = vol[:, :, z]
                unique_classes = np.unique(slice_data)
                for c in unique_classes:
                    if c < NUM_TS_CLASSES:
                        worker_stats['ts_slice_counts'][c] += 1

        except Exception as e:
            print(f"[Worker {worker_id}] Error reading TS mask {ts_path}: {e}")

    # --- 2. Process ReXGroundingCT Masks ---
    for rex_path in rex_files:
        filename = os.path.basename(rex_path)
        if filename not in rex_lookup: continue

        entry = rex_lookup[filename]
        try:
            img = nib.load(rex_path)
            vol = img.get_fdata()
            vol_shape = vol.shape

            # Calculate total theoretical voxels (H * W * D) regardless of F dimension
            if len(vol_shape) >= 3:
                worker_stats['rex_total_voxels'] += (vol_shape[-3] * vol_shape[-2] * vol_shape[-1])

            # Process 4D (F, H, W, D)
            if len(vol_shape) == 4:
                num_findings = vol_shape[0]
                depth = vol_shape[3]

                for f_idx in range(num_findings):
                    # Map the finding index to its actual class category
                    cat = entry['categories'].get(str(f_idx))
                    if cat not in REX_CLASSES: continue

                    finding_vol = vol[f_idx]

                    # Count total voxels for this specific finding
                    worker_stats['rex_voxel_counts'][cat] += np.count_nonzero(finding_vol)

                    # Count slice presence
                    for z in range(depth):
                        if np.any(finding_vol[:, :, z]):
                            worker_stats['rex_slice_counts'][cat] += 1

            # Process fallback 3D (H, W, D)
            elif len(vol_shape) == 3:
                cat = entry['categories'].get("0")
                if cat in REX_CLASSES:
                    depth = vol_shape[2]
                    worker_stats['rex_voxel_counts'][cat] += np.count_nonzero(vol)
                    for z in range(depth):
                        if np.any(vol[:, :, z]):
                            worker_stats['rex_slice_counts'][cat] += 1

        except Exception as e:
            print(f"[Worker {worker_id}] Error reading ReX mask {rex_path}: {e}")

    return worker_stats


def print_comprehensive_report(master_stats):
    """Calculates frequencies and prints a formatted report suitable for copy/pasting."""

    print("\n" + "=" * 80)
    print(" 🏥 COMPREHENSIVE MASK AUDIT REPORT")
    print("=" * 80)

    # --- TotalSegmentator Report ---
    print("\n--- TotalSegmentator (118 Classes) ---")
    ts_total = master_stats['ts_total_voxels']
    print(f"Total TS Voxels Evaluated: {ts_total:,}")

    ts_voxel_counts = master_stats['ts_voxel_counts']
    ts_slice_counts = master_stats['ts_slice_counts']

    # Calculate volumetric frequency (Base Rate)
    # Adding a small epsilon to avoid division by zero for entirely missing classes
    ts_freq = ts_voxel_counts / (ts_total + 1e-8)

    # Calculate recommended pos_weight (Inverse Frequency)
    # Clip max weight to 1000 to prevent exploding gradients for incredibly rare pixels
    ts_inverse_freq = np.clip(1.0 / (ts_freq + 1e-8), a_min=1.0, a_max=1000.0)

    print(f"\n{'Class IDX':<10} | {'Slices Present':<15} | {'Volumetric Freq':<20} | {'Rec. pos_weight':<15}")
    print("-" * 65)
    for c in range(1, NUM_TS_CLASSES):  # Skip 0 (Background)
        print(f"{c:<10} | {ts_slice_counts[c]:<15,d} | {ts_freq[c]:<20.6%} | {ts_inverse_freq[c]:<15.2f}")

    # --- ReX Report ---
    print("\n" + "=" * 80)
    print("--- ReXGroundingCT (14 Conditions) ---")
    rex_total = master_stats['rex_total_voxels']
    print(f"Total ReX Voxels Evaluated: {rex_total:,}")

    print(f"\n{'Condition':<10} | {'Slices Present':<15} | {'Volumetric Freq':<20} | {'Rec. pos_weight':<15}")
    print("-" * 65)

    rex_pos_weights = {}
    for cat in REX_CLASSES:
        voxels = master_stats['rex_voxel_counts'][cat]
        slices = master_stats['rex_slice_counts'][cat]

        freq = voxels / (rex_total + 1e-8)
        inv_freq = min(1.0 / (freq + 1e-8), 1000.0)
        rex_pos_weights[cat] = inv_freq

        print(f"{cat:<10} | {slices:<15,d} | {freq:<20.6%} | {inv_freq:<15.2f}")

    print("\n" + "=" * 80)
    print(" 🚀 READY TO PASTE INTO CONFIGURATION")
    print("=" * 80)

    # Print the array formatted for easy copy-pasting into python/yaml
    formatted_ts_weights = "[" + ", ".join([f"{w:.2f}" for w in ts_inverse_freq]) + "]"
    print("\nTotalSegmentator Recommended pos_weight array (118 dims, Index 0 is background):")
    print(formatted_ts_weights)

    formatted_rex_weights = "[" + ", ".join([f"{rex_pos_weights[cat]:.2f}" for cat in REX_CLASSES]) + "]"
    print("\nReX Recommended pos_weight array (14 dims):")
    print(formatted_rex_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_dir", required=True, help="Path to TotalSegmentator masks root")
    parser.add_argument("--rex_json", required=True, help="Path to ReXGroundingCT dataset.json")
    parser.add_argument("--rex_dir", required=True, help="Path to ReXGroundingCT segmentations root")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel processes")
    args = parser.parse_args()

    print("Scanning directories for mask files...")
    # Get all valid TS masks
    ts_lookup = scan_file_directory(args.ts_dir)
    ts_files = list(ts_lookup.values())

    # Get all valid ReX masks based on the JSON
    rex_lookup = load_rex_metadata(args.rex_json, args.rex_dir)
    rex_files = [entry['path'] for entry in rex_lookup.values() if entry['path'] is not None]

    print(f"Found {len(ts_files)} TotalSegmentator masks.")
    print(f"Found {len(rex_files)} ReXGroundingCT masks.")

    # Split workloads
    ts_chunks = np.array_split(ts_files, args.num_workers)
    rex_chunks = np.array_split(rex_files, args.num_workers)

    print(f"Starting audit with {args.num_workers} parallel workers...")

    # Use a multiprocessing pool for easy return value gathering
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.starmap(
            worker_routine,
            [(i, ts_chunks[i], rex_chunks[i], rex_lookup) for i in range(args.num_workers)]
        )

    print("Merging worker results...")
    master_stats = {
        'ts_voxel_counts': np.zeros(NUM_TS_CLASSES, dtype=np.int64),
        'ts_slice_counts': np.zeros(NUM_TS_CLASSES, dtype=np.int64),
        'ts_total_voxels': 0,
        'rex_voxel_counts': {c: 0 for c in REX_CLASSES},
        'rex_slice_counts': {c: 0 for c in REX_CLASSES},
        'rex_total_voxels': 0
    }

    for res in results:
        master_stats['ts_voxel_counts'] += res['ts_voxel_counts']
        master_stats['ts_slice_counts'] += res['ts_slice_counts']
        master_stats['ts_total_voxels'] += res['ts_total_voxels']
        master_stats['rex_total_voxels'] += res['rex_total_voxels']
        for cat in REX_CLASSES:
            master_stats['rex_voxel_counts'][cat] += res['rex_voxel_counts'][cat]
            master_stats['rex_slice_counts'][cat] += res['rex_slice_counts'][cat]

    print_comprehensive_report(master_stats)


if __name__ == "__main__":
    main()