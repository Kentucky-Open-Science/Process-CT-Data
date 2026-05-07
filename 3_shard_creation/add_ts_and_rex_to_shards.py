import argparse
import os
import glob
import json
import multiprocessing
import numpy as np
import nibabel as nib
import webdataset as wds
import cv2

# --- CONFIGURATION ---
TARGET_SIZE = (128, 128)  # Target resolution for the mask


def scan_file_directory(root_dir):
    """
    Recursively finds all .nii.gz files.
    Returns: dict { 'filename.nii.gz': '/full/path/to/filename.nii.gz' }
    """
    print(f"[Main] Scanning directory: {root_dir}...")
    lookup = {}
    pattern = os.path.join(root_dir, "**", "*.nii.gz")
    files = glob.glob(pattern, recursive=True)

    for f in files:
        name = os.path.basename(f)
        lookup[name] = f

    print(f"[Main] Found {len(lookup)} files.")
    return lookup


def load_rex_metadata(json_path, seg_dir):
    """
    Loads ReXGroundingCT dataset.json and maps filenames to findings.
    Robustly handles List, Dict of Items, or Dict of Lists (Splits).
    """
    print(f"[Main] Loading ReX metadata from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_items = []

    # --- Robust JSON Parsing ---
    if isinstance(data, list):
        all_items = data
    elif isinstance(data, dict):
        if not data:
            all_items = []
        else:
            first_val = next(iter(data.values()))
            if isinstance(first_val, list):
                # Dict of lists (e.g. {"train": [...]})
                for key, lst in data.items():
                    if isinstance(lst, list):
                        all_items.extend(lst)
            else:
                # Dict of objects
                all_items = list(data.values())

    print(f"[Main] Parsed {len(all_items)} metadata entries.")

    rex_lookup = {}
    for entry in all_items:
        if not isinstance(entry, dict): continue

        fname = entry.get('name')
        if fname:
            rex_lookup[fname] = {
                'findings': entry.get('findings', {}),
                'path': None
            }

    # Link paths
    print(f"[Main] Scanning ReX segmentation dir: {seg_dir}...")
    seg_files = glob.glob(os.path.join(seg_dir, "**", "*.nii.gz"), recursive=True)
    found_count = 0
    for f in seg_files:
        name = os.path.basename(f)
        if name in rex_lookup:
            rex_lookup[name]['path'] = f
            found_count += 1

    print(f"[Main] Matched {found_count} ReX masks to metadata entries.")
    return rex_lookup


def worker_routine(shard_paths, output_dir, ts_lookup, rex_lookup):
    """
    Processes input shards, adding both TotalSegmentator and ReX masks.
    """
    current_fname = None
    ts_vol = None
    rex_vol = None
    rex_entry = None

    # Pre-allocate blank bytes
    blank_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    _, blank_bytes = cv2.imencode('.png', blank_mask)
    blank_bytes = blank_bytes.tobytes()

    for input_path in shard_paths:
        shard_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, shard_name)

        src = wds.WebDataset(input_path).decode()
        sink = wds.TarWriter(output_path)

        try:
            for sample in src:
                # 1. Get Metadata
                meta = sample['json']
                filename = meta.get('original_file')
                slice_idx = meta.get('slice_index')

                # 2. Update Volume Caches
                if filename != current_fname:
                    current_fname = filename
                    ts_vol = None
                    rex_vol = None
                    rex_entry = None

                    # Load TotalSegmentator Volume
                    if filename and filename in ts_lookup:
                        try:
                            # Load strictly as uint8 (classes 0-117)
                            # Shape: (H, W, D)
                            ts_vol = nib.load(ts_lookup[filename]).get_fdata().astype(np.uint8)
                        except Exception as e:
                            print(f"Error loading TS mask {filename}: {e}")

                    # Load ReX Volume
                    if filename and filename in rex_lookup and rex_lookup[filename]['path']:
                        try:
                            rex_info = rex_lookup[filename]
                            rex_path = rex_info['path']
                            rex_entry = rex_info

                            # ReX Shape is (F, H, W, D) or (H, W, D)
                            rex_vol = nib.load(rex_path).get_fdata()
                        except Exception as e:
                            print(f"Error loading ReX mask {filename}: {e}")

                # 3. Process TotalSegmentator
                ts_bytes = blank_bytes
                if ts_vol is not None:
                    try:
                        # TS Shape: (H, W, D)
                        d_dim = ts_vol.shape[2]
                        if slice_idx < d_dim:
                            ts_slice = ts_vol[:, :, slice_idx]
                            ts_small = cv2.resize(ts_slice, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            _, buf = cv2.imencode('.png', ts_small)
                            ts_bytes = buf.tobytes()
                    except:
                        pass
                sample['mask.png'] = ts_bytes

                # 4. Process ReXGroundingCT (FIXED LOGIC)
                rex_bytes = blank_bytes
                rex_findings_meta = {}

                if rex_vol is not None and rex_entry is not None:
                    try:
                        vol_shape = rex_vol.shape

                        # --- CASE A: 4D Volume (F, H, W, D) ---
                        if len(vol_shape) == 4:
                            # ReX Spec: (Findings, Height, Width, Depth)
                            num_findings = vol_shape[0]
                            d_dim = vol_shape[3]

                            if slice_idx < d_dim:
                                # Start with blank accumulator
                                rex_slice_flat = np.zeros((vol_shape[1], vol_shape[2]), dtype=np.uint8)

                                for f_idx in range(num_findings):
                                    # Slice: [Finding, :, :, Slice]
                                    f_layer = rex_vol[f_idx, :, :, slice_idx]

                                    if np.any(f_layer):
                                        pixel_val = f_idx + 1
                                        rex_slice_flat[f_layer > 0] = pixel_val

                                        # Map "0" -> Text
                                        txt = rex_entry['findings'].get(str(f_idx), "")
                                        rex_findings_meta[str(pixel_val)] = txt

                                # Resize and Encode
                                rex_small = cv2.resize(rex_slice_flat, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                                _, buf = cv2.imencode('.png', rex_small)
                                rex_bytes = buf.tobytes()

                        # --- CASE B: 3D Volume (H, W, D) ---
                        elif len(vol_shape) == 3:
                            # Single finding, standard NIfTI
                            d_dim = vol_shape[2]
                            if slice_idx < d_dim:
                                f_layer = rex_vol[:, :, slice_idx]

                                if np.any(f_layer):
                                    rex_small_slice = cv2.resize(f_layer, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                                    # Pixel val 1
                                    rex_small_slice[rex_small_slice > 0] = 1

                                    txt = rex_entry['findings'].get("0", "")
                                    rex_findings_meta["1"] = txt

                                    _, buf = cv2.imencode('.png', rex_small_slice.astype(np.uint8))
                                    rex_bytes = buf.tobytes()

                    except Exception as e:
                        # print(f"Error processing ReX slice {slice_idx} in {filename}: {e}")
                        pass

                sample['rex_mask.png'] = rex_bytes
                meta['rex_findings'] = rex_findings_meta

                # Cleanup
                for k in ['rois', 'present_tissues', 'findings', 'entity_counts', 'pixels']:
                    if k in meta: del meta[k]
                sample['json'] = meta

                sink.write(sample)

        except Exception as e:
            print(f"Error processing shard {shard_name}: {e}")

        sink.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shards", required=True, help="Path to input .tar shards")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--ts_dir", required=True, help="Path to TotalSegmentator masks root")
    parser.add_argument("--rex_json", required=True, help="Path to ReXGroundingCT dataset.json")
    parser.add_argument("--rex_dir", required=True, help="Path to ReXGroundingCT segmentations root")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ts_lookup = scan_file_directory(args.ts_dir)
    rex_lookup = load_rex_metadata(args.rex_json, args.rex_dir)

    shards = sorted(glob.glob(os.path.join(args.input_shards, "*.tar")))
    print(f"[Main] Found {len(shards)} shards to process.")

    chunk_size = int(np.ceil(len(shards) / args.num_workers))
    chunks = [shards[i:i + chunk_size] for i in range(0, len(shards), chunk_size)]

    print(f"[Main] Starting {len(chunks)} workers...")

    processes = []
    for i in range(len(chunks)):
        if not chunks[i]: continue
        p = multiprocessing.Process(
            target=worker_routine,
            args=(chunks[i], args.output_dir, ts_lookup, rex_lookup)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("[Main] Done.")


if __name__ == "__main__":
    main()