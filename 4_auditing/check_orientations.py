import pandas as pd


def process_orientations(file_path, column_name, dataset_name):
    """
    Reads a CSV, cleans the orientation column, and returns value counts.
    """
    print(f"Processing {dataset_name}...")
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            print(f"  [Error] Column '{column_name}' not found.")
            return None

        # Clean and normalize the orientation strings:
        # 1. Convert to string
        # 2. Remove brackets [] and quotes '"
        # 3. Remove all spaces
        normalized = (
            df[column_name]
            .astype(str)
            .str.strip("[]'\" ")
            .str.replace(" ", "")
        )

        return normalized.value_counts()

    except FileNotFoundError:
        print(f"  [Error] File not found: {file_path}")
        return None
    except Exception as e:
        print(f"  [Error] {e}")
        return None


def main():
    # Define paths based on your metadata samples
    configs = [
        {
            "name": "RAD-ChestCT",
            "path": "dataset/RAD-ChestCT/CT_Scan_Metadata_Complete_35747.csv",
            "col": "orig_orientation"
        },
        {
            "name": "CT-RATE",
            "path": "../../heartlens/CT-RATE/dataset/metadata/train_metadata.csv",
            "col": "ImageOrientationPatient"
        }
    ]

    print("=" * 40)
    print("DATASET ORIENTATION REPORT")
    print("=" * 40)

    for config in configs:
        counts = process_orientations(config['path'], config['col'], config['name'])

        if counts is not None:
            print(f"\nUnique Orientations in {config['name']}:")
            print(f"{'Orientation (x1,y1,z1,x2,y2,z2)':<35} | {'Count':<10}")
            print("-" * 50)
            for orient, count in counts.items():
                # Display "Missing" if the value was literally "nan"
                display_orient = "Missing/Empty" if orient == "nan" else orient
                print(f"{display_orient:<35} | {count:<10}")
        print("\n" + "=" * 40)


if __name__ == "__main__":
    main()