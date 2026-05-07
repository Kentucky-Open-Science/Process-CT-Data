import os
import glob
import io
import argparse
import numpy as np
import webdataset as wds
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def decode_npy(data):
    """Decodes byte data into a numpy array."""
    return np.load(io.BytesIO(data))


def apply_stride(src, stride):
    """WebDataset pipeline filter to skip sequential slices."""
    for i, sample in enumerate(src):
        # Only yield every Nth slice
        if i % stride == 0:
            yield sample


def process_sample(sample, voxels_per_slice):
    """
    This function runs in parallel inside the DataLoader workers.
    It performs the heavy morphological operations off the main thread.
    """
    npy_data = sample.get("npy")
    if npy_data is None:
        return None

    if npy_data.ndim == 3:
        npy_data = npy_data[npy_data.shape[0] // 2]

    # Morphological Body Mask Logic
    solid_tissue = npy_data > -500
    struct = ndimage.generate_binary_structure(2, 1)
    solid_tissue = ndimage.binary_opening(solid_tissue, structure=struct, iterations=2)
    body_mask = ndimage.binary_fill_holes(solid_tissue)

    fg_voxels = npy_data[body_mask].astype(np.float64)
    num_fg = fg_voxels.size

    if num_fg == 0:
        return None

    # Calculate exact sums for global mean/std
    sum_x = float(np.sum(fg_voxels))
    sum_x2 = float(np.sum(fg_voxels ** 2))

    # Sample a subset for percentile calculations
    if num_fg > voxels_per_slice:
        sampled = np.random.choice(fg_voxels, size=voxels_per_slice, replace=False)
    else:
        sampled = fg_voxels

    # Return a dictionary of pre-calculated stats and visualization data
    return {
        "num_fg": num_fg,
        "total_voxels": npy_data.size,
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "sampled": sampled,
        "view_img": np.clip(npy_data, -1000, 400),  # Soft tissue window for plotting
        "body_mask": body_mask,
        "key": sample.get("__key__", "unknown")
    }


def filter_nones(src):
    """Removes empty samples returned by the processor."""
    for sample in src:
        if sample is not None:
            yield sample


class NPZVolumeDataset(IterableDataset):
    """
    Custom IterableDataset to read a directory of 3D .npz files.
    Splits files across dataloader workers, extracts 2D slices,
    applies stride, and runs process_sample() directly in the worker.
    """

    def __init__(self, npz_dir, stride, voxels_per_slice, npz_key=None):
        self.files = glob.glob(os.path.join(npz_dir, "*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found at {npz_dir}")
        self.stride = stride
        self.voxels_per_slice = voxels_per_slice
        self.npz_key = npz_key

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            # Split files equally among all workers
            files = np.array_split(self.files, worker_info.num_workers)[worker_info.id].tolist()

        global_slice_idx = 0
        for f in files:
            try:
                data = np.load(f)
                # Use specified key or fallback to the first array found
                key = self.npz_key if self.npz_key and self.npz_key in data else list(data.keys())[0]
                vol = data[key]

                # If volume is 3D, iterate through z-axis slices
                if vol.ndim == 3:
                    for z in range(vol.shape[0]):
                        if global_slice_idx % self.stride == 0:
                            sample = {"npy": vol[z], "__key__": f"{os.path.basename(f)}_z{z}"}
                            processed = process_sample(sample, self.voxels_per_slice)
                            if processed is not None:
                                yield processed
                        global_slice_idx += 1

                # If volume is somehow already 2D
                elif vol.ndim == 2:
                    if global_slice_idx % self.stride == 0:
                        sample = {"npy": vol, "__key__": f"{os.path.basename(f)}_flat"}
                        processed = process_sample(sample, self.voxels_per_slice)
                        if processed is not None:
                            yield processed
                    global_slice_idx += 1

            except Exception as e:
                print(f"Error loading {f}: {e}")


def main(args):
    print(f"=== Starting Parallelized CT Foreground Inspection ===")
    print(f"Input Mode: {args.input_mode.upper()}")
    print(f"Target: {args.shards_path if args.input_mode == 'wds' else args.npz_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Stride: {args.stride} (Inspecting 1 every {args.stride} slices)")
    print(f"Workers: {args.num_workers}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Data Pipeline based on input mode
    if args.input_mode == "wds":
        urls = glob.glob(args.shards_path)
        if not urls:
            raise ValueError(f"No .tar shards found at {args.shards_path}")

        dataset = (
            wds.WebDataset(urls, resampled=False, shardshuffle=True, nodesplitter=wds.split_by_node)
            .compose(wds.split_by_worker)
            .decode(wds.handle_extension("npy", decode_npy))
            .compose(lambda src: apply_stride(src, args.stride))
            .map(lambda x: process_sample(x, args.voxels_per_slice))
            .compose(filter_nones)
        )
    else:  # 'npz' mode
        dataset = NPZVolumeDataset(
            npz_dir=args.npz_dir,
            stride=args.stride,
            voxels_per_slice=args.voxels_per_slice,
            npz_key=args.npz_key
        )

    # 2. PyTorch DataLoader handles multiprocessing
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=None,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # 3. Tracking Variables
    count_total_voxels = 0
    count_fg_voxels = 0
    saved_images_count = 0

    sum_x = np.float64(0.0)
    sum_x2 = np.float64(0.0)
    reservoir_samples = []

    # 4. Processing Loop (Main Thread)
    print("\nScanning dataset...")

    for processed_count, sample in enumerate(tqdm(dataloader, total=args.max_samples, desc="Slices Aggregated")):
        if processed_count >= args.max_samples:
            break

        # Accumulate the statistics calculated by the workers
        count_fg_voxels += sample["num_fg"]
        count_total_voxels += sample["total_voxels"]
        sum_x += sample["sum_x"]
        sum_x2 += sample["sum_x2"]
        reservoir_samples.append(sample["sampled"])

        # 5. Save Visualization Samples (First 10 images)
        if saved_images_count < 10:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(sample["view_img"], cmap='gray')
            axes[0].set_title("Original CT (Windowed)")
            axes[0].axis('off')

            axes[1].imshow(sample["body_mask"], cmap='gray')
            axes[1].set_title("Morphological Body Mask")
            axes[1].axis('off')

            axes[2].imshow(sample["view_img"], cmap='gray')
            axes[2].imshow(sample["body_mask"], cmap='Reds', alpha=0.3)
            axes[2].set_title("Overlay (Red = Foreground)")
            axes[2].axis('off')

            safe_key = sample["key"].replace("/", "_").replace("\\", "_")

            plt.tight_layout()
            output_filename = os.path.join(args.output_dir, f"mask_check_{saved_images_count:02d}_{safe_key}.png")
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close()
            saved_images_count += 1

    print("\n=== Scan Complete ===")
    if count_fg_voxels == 0:
        print("No foreground voxels found. Check your thresholding logic.")
        return

    # 6. Final Calculations
    print("Calculating final statistics...")

    mean_hu = sum_x / count_fg_voxels
    variance = (sum_x2 / count_fg_voxels) - (mean_hu ** 2)
    std_hu = np.sqrt(variance)

    all_sampled_voxels = np.concatenate(reservoir_samples)
    p_0_05 = np.percentile(all_sampled_voxels, 0.05)
    p_0_5 = np.percentile(all_sampled_voxels, 0.5)
    p_99_5 = np.percentile(all_sampled_voxels, 99.5)

    fg_ratio = (count_fg_voxels / count_total_voxels) * 100

    # 7. Report
    print("\n" + "=" * 40)
    print("📊 DATASET FOREGROUND STATISTICS")
    print("=" * 40)
    print(f"Slices Analyzed      : {processed_count}")
    print(f"Total Voxels Checked : {count_total_voxels:,}")
    print(f"Foreground Voxels    : {count_fg_voxels:,} ({fg_ratio:.2f}% of total)")
    print(f"Reservoir Size       : {all_sampled_voxels.size:,} voxels used for percentiles")
    print("-" * 40)
    print(f"Exact Mean (μ)       : {mean_hu:.4f} HU")
    print(f"Exact Std (σ)        : {std_hu:.4f} HU")
    print("-" * 40)
    print(f"0.05th Percentile    : {p_0_05:.4f} HU")
    print(f"0.50th Percentile    : {p_0_5:.4f} HU  <-- (Used in TAP-CT / nnUNetv2)")
    print(f"99.50th Percentile   : {p_99_5:.4f} HU <-- (Used in TAP-CT / nnUNetv2)")
    print("=" * 40)

    print("\n💡 SUGGESTED NORMALIZATION CONFIG UPDATE:")
    print("dataset:")
    print(f"  min_hu: {p_0_5:.1f}")
    print(f"  max_hu: {p_99_5:.1f}")
    print(f"  mean_hu: {mean_hu:.4f}")
    print(f"  std_hu: {std_hu:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect CT shards or volumes to calculate global foreground statistics.")

    # Input Modes
    parser.add_argument("--input_mode", type=str, choices=["wds", "npz"], default="wds",
                        help="Choose 'wds' for WebDataset shards or 'npz' for a directory of .npz files.")
    parser.add_argument("--shards_path", type=str, default="/app/dataset/*.tar",
                        help="Glob path to the webdataset tar files (used if input_mode='wds').")
    parser.add_argument("--npz_dir", type=str, default="/app/dataset/npz",
                        help="Directory containing .npz files (used if input_mode='npz').")
    parser.add_argument("--npz_key", type=str, default=None,
                        help="Specific dictionary key to load from the .npz file (e.g. 'data'). Defaults to the first key found.")

    # Processing Params
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of 2D slices to inspect.")
    parser.add_argument("--voxels_per_slice", type=int, default=5000,
                        help="Max foreground voxels to randomly sample per slice for percentile math.")
    parser.add_argument("--stride", type=int, default=50,
                        help="Skips N slices before processing one, to ensure diverse sampling across the volume.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of CPU cores to dedicate to morphology math.")
    parser.add_argument("--output_dir", type=str, default="/workspace/outputs/mask_checks",
                        help="Writable directory to save visualization PNGs.")

    args = parser.parse_args()
    main(args)