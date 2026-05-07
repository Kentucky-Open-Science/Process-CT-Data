import argparse
import os
import glob
import json
import numpy as np
import nibabel as nib
import webdataset as wds
import cv2
import sys

# --- CONFIGURATION ---
TARGET_SIZE = (128, 128)


def scan_file_directory(root_dir):
    print(f"[Init] Scanning TS directory: {root_dir}...")
    lookup = {}
    pattern = os.path.join(root_dir, "**", "*.nii.gz")
    files = glob.glob(pattern, recursive=True)
    for f in files:
        lookup[os.path.basename(f)] = f
    return lookup


def load_rex_metadata(json_path, seg_dir):
    print(f"[Init] Loading ReX metadata from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_items = []
    if isinstance(data, list):
        all_items = data
    elif isinstance(data, dict):
        if not data:
            all_items = []
        else:
            first_val = next(iter(data.values()))
            if isinstance(first_val, list):
                for key, lst in data.items():
                    if isinstance(lst, list): all_items.extend(lst)
            else:
                all_items = list(data.values())

    rex_lookup = {}
    for entry in all_items:
        if isinstance(entry, dict):
            fname = entry.get('name')
            if fname:
                rex_lookup[fname] = {'findings': entry.get('findings', {}), 'path': None}

    print(f"[Init] Scanning ReX segmentation dir: {seg_dir}...")
    seg_files = glob.glob(os.path.join(seg_dir, "**", "*.nii.gz"), recursive=True)
    count = 0
    for f in seg_files:
        name = os.path.basename(f)
        if name in rex_lookup:
            rex_lookup[name]['path'] = f
            count += 1

    print(f"[Init] Found {count} ReX masks available.")
    return rex_lookup


def process_single_shard(input_path, output_dir, ts_lookup, rex_lookup):
    shard_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"test_{shard_name}")

    print(f"\n[Test] Processing shard: {shard_name}")
    print(f"[Test] Output will be: {output_path}")

    src = wds.WebDataset(input_path).decode()
    sink = wds.TarWriter(output_path)

    current_fname = None
    ts_vol = None
    rex_vol = None
    rex_entry = None

    processed_count = 0
    rex_hits = 0

    blank_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    _, blank_bytes = cv2.imencode('.png', blank_mask)
    blank_bytes = blank_bytes.tobytes()

    for sample in src:
        meta = sample['json']
        filename = meta.get('original_file')
        slice_idx = meta.get('slice_index')

        # Volume Loading Logic
        if filename != current_fname:
            current_fname = filename
            ts_vol = None
            rex_vol = None
            rex_entry = None

            # Load TS
            if filename in ts_lookup:
                try:
                    ts_vol = nib.load(ts_lookup[filename]).get_fdata().astype(np.uint8)
                except:
                    pass

            # Load ReX
            if filename in rex_lookup and rex_lookup[filename]['path']:
                try:
                    print(f"   -> Loading ReX Mask for: {filename}")
                    rex_info = rex_lookup[filename]
                    rex_path = rex_info['path']
                    rex_entry = rex_info
                    rex_vol = nib.load(rex_path).get_fdata()
                    # Check shape to confirm fix
                    print(f"      Shape: {rex_vol.shape}")
                except Exception as e:
                    print(f"      Error: {e}")

        # TS Processing
        ts_bytes = blank_bytes
        if ts_vol is not None:
            try:
                if slice_idx < ts_vol.shape[2]:
                    ts_slice = ts_vol[:, :, slice_idx]
                    ts_small = cv2.resize(ts_slice, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    _, buf = cv2.imencode('.png', ts_small)
                    ts_bytes = buf.tobytes()
            except:
                pass
        sample['mask.png'] = ts_bytes

        # ReX Processing (Fixed Logic)
        rex_bytes = blank_bytes
        rex_findings_meta = {}

        if rex_vol is not None:
            try:
                vol_shape = rex_vol.shape
                # Case A: 4D (F, H, W, D)
                if len(vol_shape) == 4:
                    num_findings = vol_shape[0]
                    d_dim = vol_shape[3]

                    if slice_idx < d_dim:
                        rex_slice_flat = np.zeros((vol_shape[1], vol_shape[2]), dtype=np.uint8)
                        has_content = False

                        for f_idx in range(num_findings):
                            f_layer = rex_vol[f_idx, :, :, slice_idx]
                            if np.any(f_layer):
                                rex_slice_flat[f_layer > 0] = f_idx + 1
                                txt = rex_entry['findings'].get(str(f_idx), "")
                                rex_findings_meta[str(f_idx + 1)] = txt
                                has_content = True

                        if has_content: rex_hits += 1

                        rex_small = cv2.resize(rex_slice_flat, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                        _, buf = cv2.imencode('.png', rex_small)
                        rex_bytes = buf.tobytes()

                # Case B: 3D (H, W, D)
                elif len(vol_shape) == 3:
                    if slice_idx < vol_shape[2]:
                        f_layer = rex_vol[:, :, slice_idx]
                        if np.any(f_layer):
                            rex_hits += 1
                            rex_small = cv2.resize(f_layer, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            rex_small[rex_small > 0] = 1  # Force to ID 1
                            txt = rex_entry['findings'].get("0", "")
                            rex_findings_meta["1"] = txt

                            _, buf = cv2.imencode('.png', rex_small.astype(np.uint8))
                            rex_bytes = buf.tobytes()
            except:
                pass

        sample['rex_mask.png'] = rex_bytes
        meta['rex_findings'] = rex_findings_meta

        # Cleanup
        for k in ['rois', 'present_tissues', 'findings', 'entity_counts', 'pixels']:
            if k in meta: del meta[k]
        sample['json'] = meta

        sink.write(sample)
        processed_count += 1

    sink.close()
    print(f"[Test] Finished shard. Processed {processed_count} slices.")
    print(f"[Test] ReX positive slices generated: {rex_hits}")
    return rex_hits > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shards", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ts_dir", required=True)
    parser.add_argument("--rex_json", required=True)
    parser.add_argument("--rex_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ts_lookup = scan_file_directory(args.ts_dir)
    rex_lookup = load_rex_metadata(args.rex_json, args.rex_dir)

    # Get list of shards
    shards = sorted(glob.glob(os.path.join(args.input_shards, "*.tar")))
    print(f"[Main] Found {len(shards)} shards.")

    # Iterate through shards until we find one that has a ReX file in it
    # This is inefficient but necessary to ensure we test a file that actually has ReX data
    for shard in shards:
        # Peak inside shard to check filenames without full processing?
        # Actually, let's just stream read and check the JSONs quickly.
        print(f"[Main] Checking {os.path.basename(shard)} for ReX candidates...")

        ds = wds.WebDataset(shard).decode()
        found_candidate = False
        for sample in ds:
            fname = sample['json'].get('original_file')
            if fname in rex_lookup and rex_lookup[fname]['path']:
                found_candidate = True
                break

        if found_candidate:
            print(f"[Main] !!! Found ReX candidate in {os.path.basename(shard)} !!!")
            success = process_single_shard(shard, args.output_dir, ts_lookup, rex_lookup)
            if success:
                print("\n[SUCCESS] Test shard created with ReX data.")
                print("You can now inspect it using: python inspect_shard.py test_shard.tar")
                sys.exit(0)
            else:
                print("[Warn] Processed shard but got 0 ReX hits (maybe empty slices). Trying next...")
        else:
            # print("No ReX files in this shard.")
            pass

    print("[Error] Could not find any shards containing ReX filenames.")


if __name__ == "__main__":
    main()