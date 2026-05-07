import argparse
import os
import glob
import json
import multiprocessing
import random
import webdataset as wds


def index_shard(shard_path):
    """
    Worker function for Pass 1: Reads only the JSON to count slices and flag ReX.
    Does NOT decode images, making it extremely fast.
    """
    volume_stats = {}

    # Do not call .decode() to avoid loading massive numpy/png arrays into memory
    src = wds.WebDataset(shard_path)

    for sample in src:
        # WebDataset returns raw bytes if not decoded
        if 'json' in sample:
            meta = json.loads(sample['json'])
        else:
            continue

        fname = meta.get('original_file')
        if not fname:
            continue

        # Check if this slice has ReX findings
        has_rex = False
        if meta.get('rex_findings') and len(meta['rex_findings']) > 0:
            has_rex = True

        if fname not in volume_stats:
            volume_stats[fname] = {'slice_count': 0, 'has_rex': False}

        volume_stats[fname]['slice_count'] += 1
        if has_rex:
            volume_stats[fname]['has_rex'] = True

    return volume_stats


def extraction_worker(input_shards, output_dir, selected_patients, worker_id):
    """
    Worker function for Pass 2: Decodes and writes only the selected patients.
    """
    # Create a single output tar for this worker to prevent file collision
    output_path = os.path.join(output_dir, f"subset_shard_{worker_id:04d}.tar")
    sink = wds.TarWriter(output_path)

    written_count = 0
    for shard in input_shards:
        # Now we decode because we are physically moving the images
        src = wds.WebDataset(shard).decode()
        for sample in src:
            meta = sample.get('json', {})
            fname = meta.get('original_file')

            if fname in selected_patients:
                sink.write(sample)
                written_count += 1

    sink.close()
    print(f"[Worker {worker_id}] Finished writing {written_count} slices to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shards", required=True, help="Path to input .tar shards")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--target_slices", type=int, default=500000)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.input_shards, "*.tar")))

    # ---------------------------------------------------------
    # PASS 1: Indexing all volumes across the dataset
    # ---------------------------------------------------------
    print(f"[Pass 1] Indexing {len(shards)} shards to map patients and ReX data...")
    pool = multiprocessing.Pool(args.num_workers)
    results = pool.map(index_shard, shards)

    # Merge results from all workers
    global_stats = {}
    for res in results:
        for fname, stats in res.items():
            if fname not in global_stats:
                global_stats[fname] = {'slice_count': 0, 'has_rex': False}
            global_stats[fname]['slice_count'] += stats['slice_count']
            if stats['has_rex']:
                global_stats[fname]['has_rex'] = True

    print(f"[Pass 1] Found {len(global_stats)} unique CT volumes.")

    # ---------------------------------------------------------
    # PHASE 2: Selection Logic
    # ---------------------------------------------------------
    rex_patients = [f for f, s in global_stats.items() if s['has_rex']]
    standard_patients = [f for f, s in global_stats.items() if not s['has_rex']]

    random.shuffle(standard_patients)  # Shuffle to get a random mix of standard CTs

    selected_patients = set()
    current_slice_count = 0

    # 1. Add ALL ReX patients first
    print(f"[Selection] Prioritizing {len(rex_patients)} ReX-annotated volumes...")
    for fname in rex_patients:
        selected_patients.add(fname)
        current_slice_count += global_stats[fname]['slice_count']

    # 2. Backfill with standard patients until target is reached
    print(f"[Selection] Current slices from ReX: {current_slice_count}. Backfilling to {args.target_slices}...")
    for fname in standard_patients:
        if current_slice_count >= args.target_slices:
            break
        selected_patients.add(fname)
        current_slice_count += global_stats[fname]['slice_count']

    print(f"[Selection] Final Subset: {len(selected_patients)} volumes totaling {current_slice_count} slices.")

    # ---------------------------------------------------------
    # PASS 3: Extraction and Writing
    # ---------------------------------------------------------
    print(f"[Pass 2] Extracting selected volumes to {args.output_dir}...")

    # Chunk shards for workers
    chunk_size = int(len(shards) / args.num_workers) + 1
    chunks = [shards[i:i + chunk_size] for i in range(0, len(shards), chunk_size)]

    extraction_processes = []
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        p = multiprocessing.Process(
            target=extraction_worker,
            args=(chunk, args.output_dir, selected_patients, i)
        )
        extraction_processes.append(p)
        p.start()

    for p in extraction_processes:
        p.join()

    print("[Done] Subset generation complete.")


if __name__ == "__main__":
    main()