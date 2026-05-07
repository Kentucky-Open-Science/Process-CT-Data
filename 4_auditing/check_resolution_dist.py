import sys
import csv
from collections import Counter


def main():
    # Check if the dataset path was passed as an argument
    if len(sys.argv) < 2:
        print("Usage: python check_resolution_dist.py <path_to_csv_file>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    resolution_counts = Counter()
    z_spacing_counts = Counter()
    slice_counts = Counter()

    try:
        # Open the CSV file safely
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                # --- Extract Resolution data ---
                if 'Rows' in row and 'Columns' in row:
                    rows = row.get('Rows', 'Unknown')
                    columns = row.get('Columns', 'Unknown')
                    resolution = f"{rows} x {columns}"
                elif 'orig_square' in row:
                    sq_raw = row.get('orig_square', 'Unknown')
                    try:
                        # Convert things like '512.0' to '512'
                        sq = int(float(sq_raw))
                        resolution = f"{sq} x {sq}"
                    except ValueError:
                        resolution = f"{sq_raw} x {sq_raw}"
                else:
                    resolution = "Unknown"

                resolution_counts[resolution] += 1

                # --- Extract Z-Spacing data ---
                if 'ZSpacing' in row:
                    z_spacing = row.get('ZSpacing', 'Unknown')
                elif 'orig_zdiff' in row:
                    z_spacing = row.get('orig_zdiff', 'Unknown')
                else:
                    z_spacing = "Unknown"

                z_spacing_counts[z_spacing] += 1

                # --- Extract Number of Slices data ---
                if 'NumberofSlices' in row:
                    num_slices_raw = row.get('NumberofSlices', 'Unknown')
                elif 'orig_numslices' in row:
                    num_slices_raw = row.get('orig_numslices', 'Unknown')
                else:
                    num_slices_raw = "Unknown"

                try:
                    # Convert to float first in case of decimals (e.g., '518.0'), then to int for binning
                    val = int(float(num_slices_raw))
                    bin_start = (val // 25) * 25
                    bin_end = bin_start + 24
                    bin_label = f"{bin_start}-{bin_end}"
                    slice_counts[bin_label] += 1
                except ValueError:
                    # Catch 'Unknown', missing data, or malformed strings
                    slice_counts[num_slices_raw] += 1

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    # --- Output Results ---
    header = f"--- Analysis Summary for {csv_file_path} ---"
    print("\n" + header)

    # Resolution Summary
    print("\n[ Resolution Distribution ]")
    if not resolution_counts:
        print("No resolution data found.")
    else:
        for res, count in resolution_counts.most_common():
            unit = "scan" if count == 1 else "scans"
            print(f"  {res}: {count} {unit}")

    # Helper function for histogram printing
    def print_histogram(counts, label_name):
        if not counts:
            print(f"No {label_name} data found.")
            return

        def sort_key(item):
            k = item[0]
            try:
                # If binned (e.g., "25-49"), sort by the lower bound
                if '-' in k:
                    return (0, float(k.split('-')[0]))
                return (0, float(k))
            except ValueError:
                # Push 'Unknown' or text to the bottom
                return (1, k)

        sorted_items = sorted(counts.items(), key=sort_key)
        max_count = max(counts.values())
        max_bar_length = 40  # Max character width for the histogram bar

        for label, count in sorted_items:
            unit = "scan" if count == 1 else "scans"
            bar_length = int((count / max_count) * max_bar_length) if max_count > 0 else 0
            bar = '█' * bar_length

            # Formatted to keep the pipeline aligned across different data types
            print(f"  {str(label):>12} {label_name:<7}: {count:>4} {unit:<5} | {bar}")

    # Z-Spacing Histogram
    print("\n[ Z-Spacing Distribution Histogram ]")
    print_histogram(z_spacing_counts, "spacing")

    # Number of Slices Histogram
    print("\n[ Number of Slices Histogram (Bins of 25) ]")
    print_histogram(slice_counts, "slices")

    print("\n" + "-" * len(header))


if __name__ == "__main__":
    main()