import pandas as pd
import argparse
import math
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    try:
        # Load only the column we need to save memory
        df = pd.read_csv(args.csv_path, usecols=['orig_numslices'])
        slices = df['orig_numslices'].dropna()

        # Group into bins of 50 (e.g., 123 -> 100, 160 -> 150)
        bin_size = 50
        counts = Counter((math.floor(s / bin_size) * bin_size) for s in slices)

        # Sort by bin value
        sorted_bins = sorted(counts.items())

        print(f"\nDistribution of 'orig_numslices' (Bin size: {bin_size})")
        print("-" * 50)

        # Scaling factor to keep bars within terminal width
        max_count = max(counts.values()) if counts else 1
        max_bar_width = 40

        for bin_start, count in sorted_bins:
            bin_label = f"{bin_start:4} - {bin_start + 49:4}"
            bar_len = int((count / max_count) * max_bar_width)
            bar = "#" * bar_len
            print(f"{bin_label} | {bar} ({count})")

        print("-" * 50)
        print(f"Total count: {len(slices)}\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()