import argparse
import nibabel as nib
import pandas as pd
import numpy as np
import os


def test_correction(old_vol_path, new_vol_path, metadata_path):
    print(f"--- Running Integrity Test ---")

    # 1. Load Metadata
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Extract Volume Name from path (e.g., 'train_1_a_1.nii.gz' -> 'train_1_a_1')
    vol_name = os.path.basename(old_vol_path)
    # Find row
    row = df[df['VolumeName'] == vol_name]
    if row.empty:
        raise ValueError(f"Volume {vol_name} not found in metadata!")

    slope = float(row.iloc[0]['RescaleSlope'])
    inter = float(row.iloc[0]['RescaleIntercept'])
    print(f"Found Metadata for {vol_name}: Slope={slope}, Intercept={inter}")

    # 2. Load OLD (Raw) Volume
    print(f"Loading OLD volume: {old_vol_path}")
    old_img = nib.load(old_vol_path)
    # get_fdata() usually applies header slope/inter automatically.
    # Since old header is broken (slope=1, inter=0), this returns RAW values.
    raw_data = old_img.get_fdata(dtype=np.float32)

    # Apply Manual Correction
    manual_fix = (raw_data * slope) + inter
    print(f"Applied manual correction. Range: {manual_fix.min()} to {manual_fix.max()}")

    # 3. Load NEW (Fixed) Volume
    print(f"Loading NEW volume: {new_vol_path}")
    new_img = nib.load(new_vol_path)
    # New header has correct slope/inter, so nibabel applies it automatically
    target_data = new_img.get_fdata(dtype=np.float32)
    print(f"Loaded target data. Range: {target_data.min()} to {target_data.max()}")

    # 4. Compare
    # We use allclose because floating point math might differ slightly
    match = np.allclose(manual_fix, target_data, atol=1e-3)

    if match:
        print("\n✅ SUCCESS: Manual correction matches the fixed volume exactly.")
    else:
        diff = np.abs(manual_fix - target_data).max()
        print(f"\n❌ FAILURE: Max difference was {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_vol", required=True, help="Path to the v1 (raw) nifti")
    parser.add_argument("--new_vol", required=True, help="Path to the v2 (fixed) nifti")
    parser.add_argument("--metadata", required=True, help="Path to train_metadata.csv")
    args = parser.parse_args()

    test_correction(args.old_vol, args.new_vol, args.metadata)