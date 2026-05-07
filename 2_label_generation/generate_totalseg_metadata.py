import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- TOTALSEGMENTATOR CLASSES ---
TOTALSEG_CLASSES = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
    16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
    20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left",
    24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5",
    28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
    32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
    36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5",
    40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
    44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4",
    48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart",
    52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
    55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
    79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right", 84: "gluteus_minimus_left",
    85: "gluteus_minimus_right", 86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull",
    92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7", 99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages"
}


def process_single_file(filename, dataset_dir, mask_base_dir, output_dir):
    """
    Worker function to process a single file.
    Returns an error message string if something goes wrong, or None if successful.
    """
    base_name = os.path.splitext(filename)[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")

    # Skip if already processed
    if os.path.exists(csv_path):
        return None

    # Extract subject and study assuming format like "valid_999_a_1"
    parts = base_name.split('_')
    if len(parts) >= 3:
        subject = f"{parts[0]}_{parts[1]}"
        study = f"{parts[0]}_{parts[1]}_{parts[2]}"
    else:
        return f"⚠️ Cannot parse subject/study from filename: {filename}"

    mask_path = os.path.join(mask_base_dir, subject, study, f"{base_name}.nii.gz")
    if not os.path.exists(mask_path):
        return f"⚠️ Mask not found: {mask_path}"

    # Load the .npy file strictly to get the number of slices
    npy_path = os.path.join(dataset_dir, filename)
    try:
        # mmap_mode='r' prevents loading the whole volume into memory
        vol = np.load(npy_path, mmap_mode='r')
        num_slices = vol.shape[0]
    except Exception as e:
        return f"❌ Corrupt .npy file {filename}: {e}"

    # Load NIfTI Mask
    try:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(np.int16)
    except Exception as e:
        return f"⚠️ Error loading mask {mask_path}: {e}"

    # Standard NIfTI shape is usually (X, Y, Z).
    # We need to ensure slices are on the first axis (Z, X, Y) to match the .npy file.
    if mask_data.shape[-1] == num_slices:
        mask_data = np.transpose(mask_data, (2, 0, 1))
    elif mask_data.shape[0] != num_slices:
        return f"⚠️ Dimension mismatch for {base_name}: Mask shape {mask_data.shape}, expected {num_slices} slices."

    class_ids = list(TOTALSEG_CLASSES.keys())
    class_names = list(TOTALSEG_CLASSES.values())

    # Matrix of shape (num_slices, 117)
    presence_matrix = np.zeros((num_slices, len(class_ids)), dtype=int)

    # Build presence matrix slice by slice
    for s in range(num_slices):
        unique_vals = np.unique(mask_data[s])
        for i, c_id in enumerate(class_ids):
            if c_id in unique_vals:
                presence_matrix[s, i] = 1

    # Save to CSV
    df = pd.DataFrame(presence_matrix, columns=class_names)
    df.insert(0, 'slice_id', range(num_slices))
    df.to_csv(csv_path, index=False)

    return None


def generate_metadata(dataset_dir, mask_base_dir, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)

    # Get all .npy files from the dataset directory
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npy')])

    if not files:
        print(f"⚠️ No .npy files found in {dataset_dir}")
        return

    print(f"Found {len(files)} files to process.")
    print(f"Saving metadata to: {output_dir}")
    print(f"Using {num_workers} parallel workers...")

    # Set up Multiprocessing Pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the arguments to the worker function
        futures = {
            executor.submit(process_single_file, f, dataset_dir, mask_base_dir, output_dir): f
            for f in files
        }

        # Process as they complete and update tqdm
        for future in tqdm(as_completed(futures), total=len(files), desc="Generating Metadata"):
            err_msg = future.result()
            if err_msg:
                # Use tqdm.write so it doesn't mess up the progress bar interface
                tqdm.write(err_msg)


def main():
    parser = argparse.ArgumentParser(
        description="Generate slice-level metadata from TotalSegmentator masks using Multiprocessing.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the .npy CT volumes.")
    parser.add_argument("--mask_base_dir", type=str, required=True, help="Base directory for TotalSegmentator masks.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the generated .csv metadata files will be saved.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers to use. Defaults to 8.")

    args = parser.parse_args()

    generate_metadata(args.dataset_dir, args.mask_base_dir, args.output_dir, args.num_workers)
    print("✅ Metadata generation complete.")


if __name__ == "__main__":
    main()