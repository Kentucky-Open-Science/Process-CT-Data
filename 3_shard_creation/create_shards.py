import argparse
import os
import glob
import random
import json
import time
import multiprocessing
import numpy as np
import pandas as pd
import nibabel as nib
import webdataset as wds
from tqdm import tqdm


def load_metadata_map(csv_path):
    """
    Loads the metadata CSV and returns a dictionary for O(1) lookups.
    Key: VolumeName (e.g., 'train_1_a_1.nii.gz')
    Value: {'RescaleSlope': float, 'RescaleIntercept': float}
    """
    print(f"[Main] Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Ensure keys match the filename exactly as verified in your test
    meta_map = df.set_index('VolumeName')[['RescaleSlope', 'RescaleIntercept']].to_dict('index')
    print(f"[Main] Metadata loaded for {len(meta_map)} volumes.")
    return meta_map


def load_labels_map(csv_path):
    """
    Loads the predicted labels CSV.
    Returns:
        label_map: Dict {VolumeName: [0, 1, 0, ...]} (List of ints)
        class_names: List [Class1, Class2, ...]
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"[Main] WARNING: Labels file {csv_path} not found. Shards will be created without labels.")
        return {}, []

    print(f"[Main] Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract class names (all columns after VolumeName)
    class_names = list(df.columns[1:])

    # Create dictionary: VolumeName -> List of values
    # We use apply(tolist) to ensure we get Python list of Python ints (JSON serializable)
    # instead of Numpy types which can crash the JSON serializer.
    label_map = df.set_index('VolumeName').apply(lambda row: row.tolist(), axis=1).to_dict()

    print(f"[Main] Labels loaded for {len(label_map)} volumes. Found {len(class_names)} classes.")
    return label_map, class_names


def load_blocklist(txt_path):
    """Loads the list of brain scans to ignore."""
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


def worker_routine(worker_id, file_list, metadata_map, labels_map, class_names, output_dir, max_count, max_size):
    """
    Worker function to process a subset of files and write local shards.
    """
    shard_pattern = os.path.join(output_dir, f"shards-worker{worker_id:02d}-%06d.tar")

    # Initialize ShardWriter
    sink = wds.ShardWriter(shard_pattern, maxcount=max_count, maxsize=max_size)

    processed_files = 0
    total_slices = 0
    errors = 0
    skipped_metadata = 0

    print(f"[Worker {worker_id}] Started. Processing {len(file_list)} files.")

    for nifti_path in file_list:
        filename = os.path.basename(nifti_path)

        # 1. Metadata Lookup
        if filename not in metadata_map:
            skipped_metadata += 1
            continue

        params = metadata_map[filename]
        slope = float(params['RescaleSlope'])
        inter = float(params['RescaleIntercept'])

        # 2. Label Lookup
        # We retrieve the label vector if it exists for this volume
        volume_labels = labels_map.get(filename, [])

        try:
            # 3. Load Volume
            img = nib.load(nifti_path)
            raw_data = img.get_fdata(dtype=np.float32)

            # 4. Apply Correction Only (NO CLIPPING)
            # HU = raw * slope + intercept
            hu_data = (raw_data * slope) + inter

            # 5. Cast to float16
            # CT range (-8192 to ~3000) fits safely in float16 limits (-65k to +65k)
            hu_data = hu_data.astype(np.float16)

            # 6. Slice & Write
            depth = hu_data.shape[2]

            for z in range(depth):
                slice_img = hu_data[:, :, z]

                # Metadata Sidecar
                json_payload = {
                    "original_file": filename,
                    "slice_index": z,
                    "dataset_split": "train",
                    "transform_slope": slope,
                    "transform_inter": inter,
                    "original_shape": img.shape,
                    # Added Label Info
                    "labels": volume_labels,
                    "class_names": class_names
                }

                # Write to Tar
                vol_id = filename.replace('.nii.gz', '')
                unique_key = f"{vol_id}_{z:04d}"

                sink.write({
                    "__key__": unique_key,
                    "npy": slice_img,
                    "json": json_payload
                })
                total_slices += 1

            processed_files += 1

            if processed_files % 50 == 0:
                print(
                    f"[Worker {worker_id}] Processed {processed_files}/{len(file_list)} volumes ({total_slices} slices written)...")

        except Exception as e:
            print(f"[Worker {worker_id}] ERROR processing {filename}: {e}")
            errors += 1
            continue

    sink.close()
    print(f"[Worker {worker_id}] Finished. {processed_files} OK, {skipped_metadata} Missing Meta, {errors} Errors.")


def main():
    parser = argparse.ArgumentParser(description="Parallel CT-RATE Shard Creator")
    parser.add_argument("--data_dir", required=True, help="Root directory containing raw NIfTI files")
    parser.add_argument("--metadata", required=True, help="Path to train_metadata.csv")
    parser.add_argument("--labels", required=True, help="Path to train_predicted_labels.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory for .tar shards")
    parser.add_argument("--blocklist", help="Path to no_chest_train.txt to filter brain scans")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel processes")
    parser.add_argument("--shard_size", type=int, default=5000, help="Max slices per shard")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Metadata, Blocklist, & Labels
    meta_map = load_metadata_map(args.metadata)
    labels_map, class_names = load_labels_map(args.labels)
    blocklist = load_blocklist(args.blocklist)

    # 2. Find Files
    print(f"[Main] Scanning {args.data_dir} for NIfTI files...")
    all_files = glob.glob(os.path.join(args.data_dir, '**', '*.nii.gz'), recursive=True)
    print(f"[Main] Found {len(all_files)} total files.")

    # 3. Filter & Shuffle
    valid_files = []
    print("[Main] Filtering blocklist and validating filenames...")
    for f in all_files:
        name = os.path.basename(f)
        vol_id = name.replace('.nii.gz', '')

        # Check against blocklist (both full name and ID just in case)
        if vol_id in blocklist or name in blocklist:
            continue

        valid_files.append(f)

    print(f"[Main] {len(valid_files)} files remain after filtering.")

    # Shuffle volumes so patients are mixed across shards
    random.seed(42)
    random.shuffle(valid_files)

    # 4. Split for Multiprocessing
    # Divide files into chunks for each worker
    chunk_size = len(valid_files) // args.num_workers
    chunks = [valid_files[i:i + chunk_size] for i in range(0, len(valid_files), chunk_size)]

    # Handle remainder
    if len(valid_files) % args.num_workers != 0:
        # A simpler way to chunk ensuring all files are included:
        chunks = np.array_split(valid_files, args.num_workers)

    print(f"[Main] Launching {args.num_workers} workers processing ~{len(chunks[0])} files each.")

    processes = []
    for i in range(args.num_workers):
        # Determine max_size per shard (3GB)
        p = multiprocessing.Process(
            target=worker_routine,
            args=(i, chunks[i], meta_map, labels_map, class_names, args.output_dir, args.shard_size, 3e9)
        )
        processes.append(p)
        p.start()

    # Wait for all to finish
    for p in processes:
        p.join()

    print("[Main] All workers finished. Dataset creation complete.")


if __name__ == "__main__":
    main()