import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Columns most likely to explain the size difference
INTERESTING_COLUMNS = [
    'ZSpacing',
    'SeriesDescription',
    'Rows',
    'Columns',
    'Manufacturer',
    'SliceLocation'
]


def load_metadata(csv_path):
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Ensure VolumeName is the index for fast lookup
    df['VolumeName'] = df['VolumeName'].str.strip()
    return df.set_index('VolumeName')


def get_volume_info(npy_path):
    """
    Returns (slice_count, rows, cols) using mmap to stay fast.
    """
    try:
        # mmap_mode='r' reads only the header metadata
        arr = np.load(npy_path, mmap_mode='r')
        shape = arr.shape
        # Assuming shape is (Slices, Rows, Cols)
        return shape[0], shape[1], shape[2]
    except Exception:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Compare metadata and slice dimensions of outlier vs normal volumes")
    parser.add_argument("--data_dir", required=True, help="Directory with .npy files")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--threshold", type=int, default=500, help="Slice count threshold for outliers")
    args = parser.parse_args()

    # 1. Load Metadata
    df = load_metadata(args.csv)

    # 2. Find Files
    search_path = os.path.join(args.data_dir, "*.npy")
    files = glob.glob(search_path)
    print(f"Found {len(files)} .npy files. Scanning dimensions...")

    outlier_names = []
    normal_names = []

    # 3. Scan files
    # We'll store actual dimensions found in the files to compare with CSV metadata
    file_stats = {}

    for f in tqdm(files):
        slices, rows, cols = get_volume_info(f)
        if slices is None:
            continue

        vol_name = os.path.basename(f).replace('.npy', '.nii.gz')
        file_stats[vol_name] = f"{rows}x{cols}"

        if slices > args.threshold:
            outlier_names.append(vol_name)
        else:
            normal_names.append(vol_name)

    # 4. Filter DataFrame
    df_outliers = df.reindex(outlier_names).dropna(how='all').copy()
    df_normals = df.reindex(normal_names).dropna(how='all').copy()

    # Map the actual file dimensions back to the dataframe
    df_outliers['ActualRes'] = df_outliers.index.map(file_stats)
    df_normals['ActualRes'] = df_normals.index.map(file_stats)

    print("\n" + "=" * 80)
    print(f" COMPARISON: OUTLIERS (n={len(df_outliers)}) vs NORMALS (n={len(df_normals)})")
    print("=" * 80)

    # --- METRIC 1: Z-SPACING ---
    print(f"\n[ Z-Spacing / Slice Thickness ]")
    if 'ZSpacing' in df.columns:
        out_z = pd.to_numeric(df_outliers['ZSpacing'], errors='coerce').mean()
        norm_z = pd.to_numeric(df_normals['ZSpacing'], errors='coerce').mean()
        print(f"  > Outlier Mean: {out_z:.4f} mm")
        print(f"  > Normal Mean:  {norm_z:.4f} mm")

    # --- METRIC 2: SLICE RESOLUTION (The "Size" of the slices) ---
    print(f"\n[ Slice Resolution (Rows x Columns) ]")
    print(f"{'OUTLIERS (Top Sizes)':<40} | {'NORMALS (Top Sizes)':<40}")
    print("-" * 80)

    res_out = df_outliers['ActualRes'].value_counts().head(5)
    res_norm = df_normals['ActualRes'].value_counts().head(5)

    for i in range(5):
        str_out = f"{res_out.index[i]} ({res_out.iloc[i]})" if i < len(res_out) else ""
        str_norm = f"{res_norm.index[i]} ({res_norm.iloc[i]})" if i < len(res_norm) else ""
        print(f"{str_out:<40} | {str_norm:<40}")

    # --- METRIC 3: SERIES DESCRIPTION ---
    print(f"\n[ Top Series Descriptions ]")
    if 'SeriesDescription' in df.columns:
        sd_out = df_outliers['SeriesDescription'].value_counts().head(5)
        sd_norm = df_normals['SeriesDescription'].value_counts().head(5)
        for i in range(5):
            s_out = (sd_out.index[i][:30] + "..") if i < len(sd_out) else ""
            s_norm = (sd_norm.index[i][:30] + "..") if i < len(sd_norm) else ""
            print(f"{s_out:<40} | {s_norm:<40}")

    # --- DETAILED LIST ---
    print("\n" + "=" * 80)
    print(" DETAILED OUTLIER SAMPLE (Sorted by Z-Spacing)")
    print("=" * 80)
    display_cols = ['ActualRes', 'ZSpacing', 'SeriesDescription']
    # Add any other INTERESTING_COLUMNS that exist
    display_cols += [c for c in ['Manufacturer'] if c in df.columns]

    print(df_outliers[display_cols].sort_values('ZSpacing').head(15).to_string())


if __name__ == "__main__":
    main()