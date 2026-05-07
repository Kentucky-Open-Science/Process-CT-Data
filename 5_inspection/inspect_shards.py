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
from collections import Counter

# --- CONFIGURATION ---
OUT_DIR = "inspected_samples"


def ensure_decoded(data):
    """Robustly ensures data is a valid 2D/3D numpy image array."""
    if data is None: return None
    if isinstance(data, np.ndarray) and data.ndim >= 2: return data

    try:
        if isinstance(data, np.ndarray): data = data.tobytes()
        arr = np.frombuffer(data, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return decoded
    except Exception as e:
        return None


def visualize_sample(key, scan, ts_mask, rex_mask, meta, save_path):
    """Basic visualizer to output side-by-side representations."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=100)

        # 1. Plot CT Scan
        axes[0].imshow(scan, cmap='gray')
        axes[0].set_title("Original CT Slice", fontsize=14)
        axes[0].axis('off')

        # 2. Plot TotalSegmentator Mask
        axes[1].imshow(scan, cmap='gray')
        axes[1].imshow(ts_mask, cmap='nipy_spectral', alpha=0.4)
        axes[1].set_title("TotalSegmentator Mask", fontsize=14)
        axes[1].axis('off')

        # 3. Plot ReX Mask
        axes[2].imshow(scan, cmap='gray')
        axes[2].imshow(rex_mask, cmap='jet', alpha=0.5)

        active_classes = meta.get('rex_active_classes', [])
        title_classes = ", ".join([str(c) for c in active_classes]) if active_classes else "None"
        axes[2].set_title(f"ReX Abnormalities:\n{title_classes}", fontsize=12)
        axes[2].axis('off')

        plt.suptitle(f"Sample: {key}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[Error] Failed to plot {key}: {e}")
        return False


def inspect_shards(tar_dir, num_shards_to_sample=5, visualize=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"\n[Info] Searching for samples with BOTH TotalSeg and ReX masks across {num_shards_to_sample} distinct shards...\n")

    # 1. Prioritize Dedicated ReX Shards or Fallback
    all_shards = sorted(glob.glob(os.path.join(tar_dir, "rex-shards-*.tar")))
    if not all_shards:
        all_shards = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))

    if not all_shards:
        print(f"[Error] No .tar files found in {tar_dir}")
        return

    # Filter invalid/empty shards and shuffle to get a random distribution
    shards = [s for s in all_shards if os.path.exists(s) and os.path.getsize(s) > 100]
    random.shuffle(shards)

    shards_processed = 0
    scanned_count_total = 0

    for shard_path in shards:
        shard_name = os.path.basename(shard_path)
        print(f"[Searching] Opening shard: {shard_name}...")

        # Load a single shard dataset
        dataset = wds.WebDataset([shard_path], handler=wds.warn_and_continue)

        found_in_this_shard = False

        for sample in dataset:
            scanned_count_total += 1
            key = sample['__key__']
            meta = json.loads(sample.get('json', b'{}'))
            active_classes = meta.get('rex_active_classes', [])

            # --- Check for both masks ---
            ts_mask = ensure_decoded(sample.get('mask.png'))
            has_ts = ts_mask is not None and np.max(ts_mask) > 0

            rex_mask_stacked = ensure_decoded(sample.get('rex_mask.png'))
            num_classes = len(active_classes)

            has_rex = False
            rex_mask_2d = None

            if rex_mask_stacked is not None and num_classes > 0:
                h, w = rex_mask_stacked.shape
                chunk_h = h // num_classes
                rex_mask_2d = np.zeros((chunk_h, w), dtype=np.uint8)
                for class_idx in range(num_classes):
                    layer = rex_mask_stacked[class_idx * chunk_h: (class_idx + 1) * chunk_h, :]
                    rex_mask_2d[layer > 0] = class_idx + 1
                if np.max(rex_mask_2d) > 0:
                    has_rex = True

            # If we don't have both, skip to the next slice
            if not (has_ts and has_rex):
                continue

            # Load Scan Array
            scan_raw = sample.get('npy')
            if scan_raw is None:
                continue
            scan = np.load(io.BytesIO(scan_raw)) if isinstance(scan_raw, bytes) else scan_raw

            # Resize masks back to native CT dimensions
            dsize = (scan.shape[1], scan.shape[0])
            ts_resized = cv2.resize(ts_mask, dsize, interpolation=cv2.INTER_NEAREST)
            rex_resized = cv2.resize(rex_mask_2d, dsize, interpolation=cv2.INTER_NEAREST)

            # 1. Print Metadata
            print(f"\n--- Metadata for Sample {key} (from {shard_name}) ---")
            print(json.dumps(meta, indent=2))
            print("-" * 60)

            # 2. Visualize & Save
            if visualize:
                fname = os.path.join(OUT_DIR, f"shard{shards_processed + 1}_{shard_name.replace('.tar', '')}_{key}.png")
                if visualize_sample(key, scan, ts_resized, rex_resized, meta, fname):
                    print(f"✅ [SAVED] {fname}")

            # Mark that we successfully sampled from this shard, and break out to the next shard
            found_in_this_shard = True
            break

        if found_in_this_shard:
            shards_processed += 1

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