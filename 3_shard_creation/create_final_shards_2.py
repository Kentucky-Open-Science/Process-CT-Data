import argparse
import os
import glob
import random
import json
import multiprocessing
import numpy as np
import pandas as pd
import nibabel as nib
import webdataset as wds
import cv2

TARGET_SIZE = (128, 128)


def load_metadata_map(csv_path):
    print(f"[Main] Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.fillna({'ZSpacing': 1.0})
    meta_map = df.set_index('VolumeName')[['RescaleSlope', 'RescaleIntercept', 'ZSpacing']].to_dict('index')
    print(f"[Main] Metadata loaded for {len(meta_map)} volumes.")
    return meta_map


def load_labels_map(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}, []
    df = pd.read_csv(csv_path)
    class_names = list(df.columns[1:])
    label_map = df.set_index('VolumeName').apply(lambda row: row.tolist(), axis=1).to_dict()
    return label_map, class_names


def load_blocklist(txt_path):
    blocklist = set()
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                blocklist.add(os.path.basename(line.strip()).replace('.nii.gz', ''))
    return blocklist


def scan_file_directory(root_dir):
    lookup = {}
    if root_dir and os.path.exists(root_dir):
        files = glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
        for f in files:
            lookup[os.path.basename(f)] = f
    return lookup


def load_rex_metadata(json_path, seg_dir):
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
                'findings': entry.get('findings', {}),
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


def worker_routine(worker_id, file_list, meta_map, labels_map, class_names, ts_lookup, rex_lookup, output_dir,
                   max_count):
    # Separate counters and writers for ReX and Standard shards
    rex_shard_idx, std_shard_idx = 0, 0
    current_rex_slice_count, current_std_slice_count = 0, 0

    # Half the shard size for ReX to account for the stacked PNGs
    rex_max_count = max(1, max_count // 2)
    std_max_count = max_count

    rex_sink, current_rex_tar = None, None
    std_sink, current_std_tar = None, None

    def get_sink(is_rex, s_idx):
        prefix = "rex-shards" if is_rex else "shards"
        tar_name = f"{prefix}-w{worker_id:02d}-{s_idx:06d}.tar"
        return wds.TarWriter(os.path.join(output_dir, tar_name)), tar_name

    worker_scan_lookup = {}
    blank_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    _, blank_bytes_enc = cv2.imencode('.png', blank_mask)
    blank_bytes = blank_bytes_enc.tobytes()

    for nifti_path in file_list:
        filename = os.path.basename(nifti_path)
        if filename not in meta_map: continue

        params = meta_map[filename]
        slope, inter = float(params['RescaleSlope']), float(params['RescaleIntercept'])
        z_spacing = float(params['ZSpacing'])
        volume_labels = labels_map.get(filename, [])

        is_rex_scan = filename in rex_lookup and rex_lookup[filename]['path']

        try:
            # 1. Load Main CT
            img = nib.load(nifti_path)
            raw_data = img.get_fdata(dtype=np.float32)
            hu_data = ((raw_data * slope) + inter).astype(np.float16)
            depth = hu_data.shape[2]

            # 2. Check Shard Rollover & Lazy Initialize Sinks
            if is_rex_scan:
                if rex_sink is None:
                    rex_sink, current_rex_tar = get_sink(True, rex_shard_idx)
                elif current_rex_slice_count + depth > rex_max_count and current_rex_slice_count > 0:
                    rex_sink.close()
                    rex_shard_idx += 1
                    rex_sink, current_rex_tar = get_sink(True, rex_shard_idx)
                    current_rex_slice_count = 0
                worker_scan_lookup[filename] = current_rex_tar
                active_sink = rex_sink
            else:
                if std_sink is None:
                    std_sink, current_std_tar = get_sink(False, std_shard_idx)
                elif current_std_slice_count + depth > std_max_count and current_std_slice_count > 0:
                    std_sink.close()
                    std_shard_idx += 1
                    std_sink, current_std_tar = get_sink(False, std_shard_idx)
                    current_std_slice_count = 0
                worker_scan_lookup[filename] = current_std_tar
                active_sink = std_sink

            # 3. Load Masks
            ts_vol, rex_vol, rex_entry = None, None, None
            if filename in ts_lookup:
                ts_vol = nib.load(ts_lookup[filename]).get_fdata().astype(np.uint8)
            if is_rex_scan:
                rex_entry = rex_lookup[filename]
                rex_vol = nib.load(rex_entry['path']).get_fdata()

            # 4. Process Slices
            for z in range(depth):
                slice_img = hu_data[:, :, z]

                # TS Mask Processing
                ts_bytes = blank_bytes
                if ts_vol is not None and z < ts_vol.shape[2]:
                    ts_small = cv2.resize(ts_vol[:, :, z], TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode('.png', ts_small)
                    ts_bytes = buf.tobytes()

                # ReX Mask Processing (Stacked PNG)
                rex_bytes = blank_bytes
                active_categories = []

                if rex_vol is not None and rex_entry is not None:
                    vol_shape = rex_vol.shape
                    active_masks = []

                    if len(vol_shape) == 4 and z < vol_shape[3]:  # (F, H, W, D)
                        num_findings = vol_shape[0]
                        for f_idx in range(num_findings):
                            f_layer = rex_vol[f_idx, :, :, z]
                            if np.any(f_layer):
                                layer_small = cv2.resize(f_layer, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                                layer_small[layer_small > 0] = 255  # Convert to 8-bit binary
                                active_masks.append(layer_small.astype(np.uint8))
                                active_categories.append(rex_entry['categories'].get(str(f_idx), ""))

                    elif len(vol_shape) == 3 and z < vol_shape[2]:  # (H, W, D) Fallback
                        f_layer = rex_vol[:, :, z]
                        if np.any(f_layer):
                            layer_small = cv2.resize(f_layer, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            layer_small[layer_small > 0] = 255
                            active_masks.append(layer_small.astype(np.uint8))
                            active_categories.append(rex_entry['categories'].get("0", ""))

                    # Stack and encode if any abnormalities exist in this slice
                    if active_masks:
                        stacked_mask = np.vstack(active_masks)
                        _, buf = cv2.imencode('.png', stacked_mask)
                        rex_bytes = buf.tobytes()

                # Construct JSON Payload
                json_payload = {
                    "original_file": filename,
                    "slice_index": z,
                    "dataset_split": "train",
                    "transform_slope": slope,
                    "transform_inter": inter,
                    "z_spacing": z_spacing,
                    "original_shape": img.shape,
                    "labels": volume_labels,
                    "class_names": class_names,
                    "rex_active_classes": active_categories  # Ordered list matching the vertical stack
                }

                vol_id = filename.replace('.nii.gz', '')
                unique_key = f"{vol_id}_{z:04d}"

                active_sink.write({
                    "__key__": unique_key,
                    "npy": slice_img,
                    "mask.png": ts_bytes,
                    "rex_mask.png": rex_bytes,
                    "json": json_payload
                })

            # Update respective slice counts
            if is_rex_scan:
                current_rex_slice_count += depth
            else:
                current_std_slice_count += depth

        except Exception as e:
            print(f"[Worker {worker_id}] ERROR processing {filename}: {e}")

    # Close any open sinks
    if rex_sink: rex_sink.close()
    if std_sink: std_sink.close()

    lookup_path = os.path.join(output_dir, f"worker_{worker_id}_lookup.json")
    with open(lookup_path, 'w') as f:
        json.dump(worker_scan_lookup, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root directory containing raw NIfTI files")
    parser.add_argument("--metadata", required=True, help="Path to train_metadata.csv")
    parser.add_argument("--labels", required=True, help="Path to train_predicted_labels.csv")
    parser.add_argument("--ts_dir", default="", help="Path to TotalSegmentator masks root")
    parser.add_argument("--rex_json", default="", help="Path to ReXGroundingCT dataset.json")
    parser.add_argument("--rex_dir", default="", help="Path to ReXGroundingCT segmentations")
    parser.add_argument("--output_dir", required=True, help="Output directory for .tar shards")
    parser.add_argument("--blocklist", help="Path to no_chest_train.txt")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--shard_size", type=int, default=5000, help="Max slices per shard")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    meta_map = load_metadata_map(args.metadata)
    labels_map, class_names = load_labels_map(args.labels)
    blocklist = load_blocklist(args.blocklist)
    ts_lookup = scan_file_directory(args.ts_dir)
    rex_lookup = load_rex_metadata(args.rex_json, args.rex_dir)

    all_files = glob.glob(os.path.join(args.data_dir, '**', '*.nii.gz'), recursive=True)
    valid_files = [f for f in all_files if
                   os.path.basename(f).replace('.nii.gz', '') not in blocklist and os.path.basename(f) not in blocklist]

    random.seed(42)
    random.shuffle(valid_files)

    chunks = np.array_split(valid_files, args.num_workers)

    processes = []
    for i in range(args.num_workers):
        p = multiprocessing.Process(
            target=worker_routine,
            args=(i, chunks[i], meta_map, labels_map, class_names, ts_lookup, rex_lookup, args.output_dir,
                  args.shard_size)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("[Main] Merging worker lookup files...")
    master_lookup = {}
    for i in range(args.num_workers):
        lookup_path = os.path.join(args.output_dir, f"worker_{i}_lookup.json")
        if os.path.exists(lookup_path):
            with open(lookup_path, 'r') as f:
                worker_dict = json.load(f)
                master_lookup.update(worker_dict)
            os.remove(lookup_path)

    final_lookup_path = os.path.join(args.output_dir, "scan_to_shard_lookup.json")
    with open(final_lookup_path, 'w') as f:
        json.dump(master_lookup, f, indent=4)

    print(f"[Main] Complete! Saved master lookup to {final_lookup_path}")


if __name__ == "__main__":
    main()