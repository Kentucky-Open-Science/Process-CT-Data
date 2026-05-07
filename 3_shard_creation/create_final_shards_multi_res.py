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

    missing_mask = df[['XYSpacing', 'ZSpacing']].isna().any(axis=1)
    missing_count = missing_mask.sum()
    if missing_count > 0:
        print(f"[Main] ⚠️ Skipping {missing_count} volumes due to missing XYSpacing or ZSpacing.")
        df = df[~missing_mask]

    meta_map = df.set_index('VolumeName')[['RescaleSlope', 'RescaleIntercept', 'ZSpacing', 'XYSpacing']].to_dict(
        'index')
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


def worker_routine(worker_id, group_list, meta_map, labels_map, class_names, ts_lookup, rex_lookup, output_dir,
                   max_count):
    shard_idx = 0
    current_slice_count = 0
    sink, current_tar = None, None

    def get_sink(s_idx):
        tar_name = f"shards-w{worker_id:02d}-{s_idx:06d}.tar"
        return wds.TarWriter(os.path.join(output_dir, tar_name)), tar_name

    worker_scan_lookup = {}
    blank_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    _, blank_bytes_enc = cv2.imencode('.png', blank_mask)
    blank_bytes = blank_bytes_enc.tobytes()

    for base_scan_id, file_list in group_list:
        valid_files = [f for f in file_list if os.path.basename(f) in meta_map]
        if not valid_files: continue

        # Sort by ZSpacing descending to find the anchor (Lowest Resolution)
        valid_files.sort(key=lambda f: float(meta_map[os.path.basename(f)]['ZSpacing']), reverse=True)
        anchor_file = valid_files[0]
        anchor_name = os.path.basename(anchor_file)
        anchor_z_spacing = float(meta_map[anchor_name]['ZSpacing'])

        try:
            # 1. Load all volumes in the group into memory
            volumes_data = {}
            for f in valid_files:
                name = os.path.basename(f)
                img = nib.load(f)
                params = meta_map[name]
                slope, inter = float(params['RescaleSlope']), float(params['RescaleIntercept'])

                # Parse XYSpacing
                xy_raw = params['XYSpacing']
                if isinstance(xy_raw, str):
                    xy_clean = xy_raw.strip('[]').replace("'", "").split(',')
                    x_spacing = float(xy_clean[0])
                    y_spacing = float(xy_clean[1]) if len(xy_clean) > 1 else x_spacing
                else:
                    x_spacing = y_spacing = float(xy_raw)

                hu_data = ((img.get_fdata(dtype=np.float32) * slope) + inter).astype(np.float16)

                # Check for masks assigned to this specific file
                ts_vol = nib.load(ts_lookup[name]).get_fdata().astype(np.int16) if name in ts_lookup else None

                rex_vol, rex_entry = None, None
                if name in rex_lookup and rex_lookup[name]['path']:
                    rex_entry = rex_lookup[name]
                    rex_vol = nib.load(rex_entry['path']).get_fdata().astype(np.int16)

                volumes_data[name] = {
                    'clean_name': name.replace('.nii.gz', ''),
                    'hu': hu_data,
                    'z_spacing': float(params['ZSpacing']),
                    'xy_spacing': [x_spacing, y_spacing],
                    'labels': labels_map.get(name, []),
                    'ts_vol': ts_vol,
                    'rex_vol': rex_vol,
                    'rex_entry': rex_entry
                }

            anchor_depth = volumes_data[anchor_name]['hu'].shape[2]

            # 2. Check Shard Rollover
            if sink is None:
                sink, current_tar = get_sink(shard_idx)
            elif current_slice_count + anchor_depth > max_count and current_slice_count > 0:
                sink.close()
                shard_idx += 1
                sink, current_tar = get_sink(shard_idx)
                current_slice_count = 0

            worker_scan_lookup[base_scan_id] = current_tar

            # 3. Process Slices aligned to the Anchor's physical depth
            for z_anchor in range(anchor_depth):
                physical_z = z_anchor * anchor_z_spacing

                unique_key = f"{base_scan_id}_{z_anchor:04d}"
                wds_payload = {"__key__": unique_key}

                json_payload = {
                    "base_scan_id": base_scan_id,
                    "anchor_file": anchor_name,
                    "anchor_slice_index": z_anchor,
                    "physical_z_depth_mm": physical_z,
                    "class_names": class_names,
                    "reconstructions": {},
                    "rex_active_classes": {}
                }

                # 4. Extract nearest slices for all reconstructions
                for name, v_data in volumes_data.items():
                    clean_name = v_data['clean_name']

                    # Nearest Neighbor Match
                    z_target = int(round(physical_z / v_data['z_spacing']))
                    z_target = min(v_data['hu'].shape[2] - 1, max(0, z_target))

                    # Store Native Array
                    wds_payload[f"{clean_name}.npy"] = v_data['hu'][:, :, z_target]

                    # Store Metadata for this specific reconstruction
                    json_payload["reconstructions"][clean_name] = {
                        "native_z_index": z_target,
                        "original_z_spacing": v_data['z_spacing'],
                        "original_xy_spacing": v_data['xy_spacing'],
                        "original_shape": v_data['hu'].shape,
                        "labels": v_data['labels']
                    }

                    # Extract TotalSegmentator Mask
                    if v_data['ts_vol'] is not None and z_target < v_data['ts_vol'].shape[2]:
                        ts_small = cv2.resize(v_data['ts_vol'][:, :, z_target].astype(np.uint8), TARGET_SIZE,
                                              interpolation=cv2.INTER_NEAREST)
                        _, buf = cv2.imencode('.png', ts_small)
                        wds_payload[f"{clean_name}_ts.png"] = buf.tobytes()

                    # Extract ReXGroundingCT Mask
                    if v_data['rex_vol'] is not None and v_data['rex_entry'] is not None:
                        active_categories = []
                        active_masks = []
                        vol_shape = v_data['rex_vol'].shape

                        if len(vol_shape) == 4 and z_target < vol_shape[3]:
                            num_findings = vol_shape[0]
                            for f_idx in range(num_findings):
                                f_layer = v_data['rex_vol'][f_idx, :, :, z_target]
                                if np.any(f_layer):
                                    layer_small = cv2.resize(f_layer.astype(np.uint8), TARGET_SIZE,
                                                             interpolation=cv2.INTER_NEAREST)
                                    layer_small[layer_small > 0] = 255
                                    active_masks.append(layer_small)
                                    active_categories.append(v_data['rex_entry']['categories'].get(str(f_idx), ""))

                        if active_masks:
                            stacked_mask = np.vstack(active_masks)
                            _, buf = cv2.imencode('.png', stacked_mask)
                            wds_payload[f"{clean_name}_rex.png"] = buf.tobytes()
                            json_payload["rex_active_classes"][clean_name] = active_categories

                wds_payload["json"] = json_payload
                sink.write(wds_payload)

            current_slice_count += anchor_depth

        except Exception as e:
            print(f"[Worker {worker_id}] ERROR processing group {base_scan_id}: {e}")

    if sink: sink.close()

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
    parser.add_argument("--shard_size", type=int, default=1000, help="Max lowest-res slices per shard")
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

    print(f"[Main] Initial valid files found: {len(valid_files)}")

    # --- GROUPING LOGIC ---
    scan_groups = {}
    for f in valid_files:
        basename = os.path.basename(f).replace('.nii.gz', '')
        parts = basename.split('_')
        base_scan_id = "_".join(parts[:3]) if len(parts) >= 3 else basename
        scan_groups.setdefault(base_scan_id, []).append(f)

    group_list = list(scan_groups.items())
    print(f"[Main] Total unique patient/study groups to process: {len(group_list)}")

    random.seed(42)
    random.shuffle(group_list)

    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    chunks = chunkify(group_list, args.num_workers)

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
                master_lookup.update(json.load(f))
            os.remove(lookup_path)

    final_lookup_path = os.path.join(args.output_dir, "scan_to_shard_lookup.json")
    with open(final_lookup_path, 'w') as f:
        json.dump(master_lookup, f, indent=4)

    print(f"[Main] Complete! Saved master lookup to {final_lookup_path}")


if __name__ == "__main__":
    main()



'''
The New WebDataset StructureThe resulting .tar files now contain grouped representations of a single physical coordinate rather than a single file structure. Because you are keeping native spacing, the arrays inside the same sample may have different dimensions (e.g., one array is 512x512, the other is 768x768).Here is what a single slice index inside the .tar file looks like:Sample Key: train_1_a_0042 (Base ID + Anchor Z-Index)train_1_a_1.npy: The high-resolution native HU array at native matching index (e.g., $z=120$).train_1_a_2.npy: The low-resolution native HU array at anchor index (e.g., $z=42$).train_1_a_1_ts.png: The TotalSegmentator mask, specifically mapped to train_1_a_1.npy's coordinates.train_1_a_2_rex.png: The ReX mask, specifically mapped to train_1_a_2.npy's coordinates.json: A master metadata map for the slice.The json object structurally maps how the arrays align. It provides the anchor's physical depth and the specific configurations of the reconstructions included in that sample:
{
  "base_scan_id": "train_1_a",
  "anchor_file": "train_1_a_2.nii.gz",
  "anchor_slice_index": 42,
  "physical_z_depth_mm": 210.0,
  "class_names": ["Cardiomegaly", "Nodule", "..."],
  "reconstructions": {
    "train_1_a_1": {
      "native_z_index": 120,
      "original_z_spacing": 1.75,
      "original_xy_spacing": [0.8, 0.8],
      "original_shape": [512, 512, 300],
      "labels": [1, 0, "..."]
    },
    "train_1_a_2": {
      "native_z_index": 42,
      "original_z_spacing": 5.0,
      "original_xy_spacing": [0.8, 0.8],
      "original_shape": [512, 512, 85],
      "labels": [1, 0, "..."]
    }
  },
  "rex_active_classes": {
    "train_1_a_2": ["Atelectasis"]
  }
}
'''