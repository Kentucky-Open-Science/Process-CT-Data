import pandas as pd
import glob


def main():
    print("Loading CSV files...")
    # 1. Load and combine the 3 split files
    files = [
        "./dataset/RAD-ChestCT/imgtrain_Abnormality_and_Location_Labels.csv",
        "./dataset/RAD-ChestCT/imgvalid_Abnormality_and_Location_Labels.csv",
        "./dataset/RAD-ChestCT/imgtest_Abnormality_and_Location_Labels.csv"
    ]

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
            print(f"Loaded {f}")
        except FileNotFoundError:
            print(f"File not found: {f}. Make sure you are in the correct directory.")

    if not dfs:
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined_df)}")

    # 2. Define the exact 14-class mapping based on the CT-RATE paper methodology
    mapping = {
        "calcification": "Calcification",
        "cardiomegaly": "Cardiomegaly",
        "pericardial_effusion": "Pericardial effusion",
        "hernia": "Hiatal hernia",
        "emphysema": "Emphysema",
        "atelectasis": "Atelectasis",
        "nodule": "Lung nodule",  # You already handle the nodulegr1cm logic below
        "opacity": "Lung opacity",
        "fibrosis": "Pulmonary fibrotic sequela",
        "pleural_effusion": "Pleural effusion",
        "bronchial_wall_thickening": "Peribronchial thickening",
        "consolidation": "Consolidation",
        "bronchiectasis": "Bronchiectasis",
        "septal_thickening": "Interlobular septal thickening",
        "lymphadenopathy": "Lymphadenopathy",
        "medical_material": "Medical material"  # We will handle this custom prefix list below
    }

    # 3. Create a new dataframe to hold the condensed features
    condensed_df = pd.DataFrame()
    condensed_df["NoteAcc_DEID"] = combined_df["NoteAcc_DEID"]

    # 4. Collapse locations for each base abnormality
    print("Condensing location columns into the 16 evaluation classes...")
    for rad_name, ctrate_name in mapping.items():
        if rad_name == "nodule":
            cols_to_combine = [col for col in combined_df.columns if
                               col.startswith("nodule*") or col.startswith("nodulegr1cm*")]
        elif rad_name == "medical_material":
            # Combine all device-related prefixes
            device_prefixes = ("catheter_or_port*", "chest_tube*", "clip*", "gi_tube*",
                               "hardware*", "pacemaker_or_defib*", "stent*", "tracheal_tube*")
            cols_to_combine = [col for col in combined_df.columns if col.startswith(device_prefixes)]
        else:
            cols_to_combine = [col for col in combined_df.columns if col.startswith(f"{rad_name}*")]

        if cols_to_combine:
            condensed_df[ctrate_name] = combined_df[cols_to_combine].max(axis=1)
        else:
            print(f"  -> Warning: No columns found for prefix '{rad_name}'")

    # 5. Save the result
    output_file = "./dataset/RAD-ChestCT/rad_labels.csv"
    condensed_df.to_csv(output_file, index=False)
    print(f"\nDone! Condensed labels saved to {output_file}")

    # 6. Print class distribution summary
    print("\n" + "=" * 45)
    print("📊 FINAL CT-RATE EVALUATION CLASSES")
    print("=" * 45)
    class_counts = condensed_df.drop(columns=["NoteAcc_DEID"]).sum().sort_values(ascending=False)

    for class_name, count in class_counts.items():
        print(f"{class_name:<30} | {int(count):>6} cases")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()