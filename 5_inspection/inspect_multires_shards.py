import argparse
import os
import glob
import json
import random
import numpy as np
import webdataset as wds
import matplotlib.pyplot as plt
import cv2
import io

# --- CONFIGURATION ---
OUT_DIR = "inspected_multiplexed_samples"


def ensure_decoded(data):
    """Robustly ensures data is a valid 2D/3D numpy image array."""
    if data is None: return None
    if isinstance(data, np.ndarray) and data.ndim >= 2: return data

    try:
        if isinstance(data, bytes):
            arr = np.frombuffer(data, np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            return decoded
    except Exception as e:
        return None


def visualize_multiplexed_sample(key, recons_data, meta, save_path):
    """Visualizes all reconstructions for a given physical Z-depth."""
    num_recons = len(recons_data)
    if num_recons == 0:
        return False

    try:
        # Create an N x 3 grid (Rows: Reconstructions, Cols: CT, TS, ReX)
        fig, axes = plt.subplots(num_recons, 3, figsize=(15, 5 * num_recons), dpi=100)

        # Ensure axes is a 2D array even if there's only one reconstruction
        if num_recons == 1:
            axes = np.expand_dims(axes, axis=0)

        physical_z = meta.get('physical_z_depth_mm', 'Unknown')

        for row, (recon_name, data) in enumerate(recons_data.items()):
            scan = data['scan']
            ts_mask = data['ts_mask']
            rex_mask = data['rex_mask']
            recon_meta = meta['reconstructions'][recon_name]

            native_z = recon_meta.get('native_z_index', 'Unknown')
            z_spacing = recon_meta.get('original_z_spacing', 'Unknown')

            dsize = (scan.shape[1], scan.shape[0])

            # 1. Plot CT Scan
            axes[row, 0].imshow(scan, cmap='gray')
            axes[row, 0].set_title(f"{recon_name}\nNative Z: {native_z} | Spacing: {z_spacing}mm", fontsize=12)
            axes[row, 0].axis('off')

            # 2. Plot TotalSegmentator Mask
            axes[row, 1].imshow(scan, cmap='gray')
            if ts_mask is not None:
                ts_resized = cv2.resize(ts_mask, dsize, interpolation=cv2.INTER_NEAREST)
                axes[row, 1].imshow(ts_resized, cmap='nipy_spectral', alpha=0.4)
                axes[row, 1].set_title("TotalSegmentator Mask", fontsize=12)
            else:
                axes[row, 1].set_title("No TS Mask", fontsize=12, color='gray')
            axes[row, 1].axis('off')

            # 3. Plot ReX Mask
            axes[row, 2].imshow(scan, cmap='gray')
            if rex_mask is not None:
                rex_resized = cv2.resize(rex_mask, dsize, interpolation=cv2.INTER_NEAREST)
                axes[row, 2].imshow(rex_resized, cmap='jet', alpha=0.5)

                # Fetch active classes specifically for this reconstruction
                active_classes = meta.get('rex_active_classes', {}).get(recon_name, [])
                title_classes = ", ".join([str(c) for c in active_classes]) if active_classes else "Unknown"
                axes[row, 2].set_title(f"ReX Abnormalities:\n{title_classes}", fontsize=12)
            else:
                axes[row, 2].set_title("No ReX Mask", fontsize=12, color='gray')
            axes[row, 2].axis('off')

        plt.suptitle(f"Sample: {key} | Anchor Depth: {physical_z}mm", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[Error] Failed to plot {key}: {e}")
        return False


def inspect_shards(tar_dir, num_shards_to_sample=5, visualize=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n[Info] Inspecting multiplexed shards for random slices containing ReX Masks...\n")

    all_shards = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    if not all_shards:
        print(f"[Error] No .tar files found in {tar_dir}")
        return

    shards = [s for s in all_shards if os.path.exists(s) and os.path.getsize(s) > 100]
    random.shuffle(shards)

    shards_processed = 0
    scanned_count_total = 0

    for shard_path in shards:
        shard_name = os.path.basename(shard_path)
        print(f"[Searching] Scanning {shard_name} for valid ReX slices...")

        dataset = wds.WebDataset([shard_path], handler=wds.warn_and_continue)

        # Reservoir sampling variables
        selected_sample_data = None
        valid_rex_count = 0

        for sample in dataset:
            scanned_count_total += 1
            key = sample['__key__']

            # Extract main JSON payload
            meta_raw = sample.get('json')
            if not meta_raw:
                continue
            meta = json.loads(meta_raw) if isinstance(meta_raw, bytes) else meta_raw

            if 'reconstructions' not in meta:
                continue

            recons_data = {}
            has_rex_mask = False

            # Iterate through all multiplexed reconstructions for this physical slice
            for recon_name in meta['reconstructions'].keys():

                # 1. Extract Native Array
                npy_key = f"{recon_name}.npy"
                scan_raw = sample.get(npy_key)
                if scan_raw is None:
                    continue
                scan = np.load(io.BytesIO(scan_raw)) if isinstance(scan_raw, bytes) else scan_raw

                # 2. Extract TotalSegmentator Mask
                ts_key = f"{recon_name}_ts.png"
                ts_mask = ensure_decoded(sample.get(ts_key))

                # 3. Extract and unstack ReX Mask
                rex_key = f"{recon_name}_rex.png"
                rex_mask_stacked = ensure_decoded(sample.get(rex_key))
                rex_mask_2d = None

                active_classes = meta.get('rex_active_classes', {}).get(recon_name, [])
                num_classes = len(active_classes)

                if rex_mask_stacked is not None and num_classes > 0:
                    h, w = rex_mask_stacked.shape
                    chunk_h = h // num_classes
                    rex_mask_2d = np.zeros((chunk_h, w), dtype=np.uint8)
                    for class_idx in range(num_classes):
                        layer = rex_mask_stacked[class_idx * chunk_h: (class_idx + 1) * chunk_h, :]
                        rex_mask_2d[layer > 0] = class_idx + 1

                    if np.max(rex_mask_2d) > 0:
                        has_rex_mask = True

                recons_data[recon_name] = {
                    'scan': scan,
                    'ts_mask': ts_mask,
                    'rex_mask': rex_mask_2d
                }

            # If this physical slice contains at least one ReX mask, apply Reservoir Sampling
            if has_rex_mask:
                valid_rex_count += 1
                # Replace the currently held sample with probability 1/valid_rex_count
                if random.random() < (1.0 / valid_rex_count):
                    selected_sample_data = (key, recons_data, meta)

        # After finishing the shard, plot the randomly selected sample
        if selected_sample_data is not None:
            final_key, final_recons, final_meta = selected_sample_data

            print(f"\n--- Metadata for Random ReX Sample {final_key} (from {shard_name}) ---")
            print(json.dumps(final_meta, indent=2))
            print("-" * 60)

            if visualize:
                fname = os.path.join(OUT_DIR,
                                     f"shard{shards_processed + 1}_{shard_name.replace('.tar', '')}_{final_key}.png")
                if visualize_multiplexed_sample(final_key, final_recons, final_meta, fname):
                    print(f"✅ [SAVED] {fname} (Selected from {valid_rex_count} valid slices in shard)")

            shards_processed += 1
        else:
            print(f"[-] No ReX masks found in {shard_name}.")

        if shards_processed >= num_shards_to_sample:
            break

    print(f"\n[Done] Successfully sampled from {shards_processed} shards. Scanned {scanned_count_total} total slices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tar_dir", help="Directory containing .tar shards")
    parser.add_argument("--num_shards", type=int, default=10,
                        help="Number of different .tar files to extract a sample from")
    parser.add_argument("--visualize", type=str, default="true", help="Set to 'false' to only print metadata")

    args = parser.parse_args()
    viz_bool = args.visualize.lower() == "true"

    inspect_shards(args.tar_dir, args.num_shards, viz_bool)