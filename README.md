# Process-CT-Data

> **Paper Repository** — Preprocessing pipeline for converting raw 3D CT volumes into a slice-based WebDataset format for scalable distributed training.

---

## Overview

This repository implements the dataset preparation pipeline described in the paper. Raw 3D CT volumes and annotations from the CT-RATE dataset are converted into a slice-based [WebDataset](https://github.com/webdataset/webdataset) format optimized for high-throughput, multi-GPU training. The dataset is structured to natively support 2D, 2.5D (slab), and 3D training paradigms — the actual sampling logic for each modality (e.g., 2.5D slab extraction, 3D patch sampling) lives in the dataloader of the training codebase, but the shards produced here are pre-organized to make those lookups fast and efficient. The pipeline handles:

1. **HU Conversion & Float16 Casting** — Raw NIfTI volumes are converted to Hounsfield Units (HU) using per-volume Rescale Slope/Intercept metadata and cast to 16-bit floating-point precision.

2. **Dual Annotation Synchronization** — Two annotation sources are spatially aligned:
   - **TotalSegmentator (TS):** 118 anatomical classes, automatically generated for all images, resized to 128×128.
   - **ReXGroundingCT (ReX):** Professional annotations for multi-label radiological findings (14 classes), binarized and resized to 128×128.

3. **Slab-Based Grouping** — Incoming slices are grouped into continuous 12 mm slabs, dynamically calculated using Z-spacing metadata, enabling efficient 2.5D sampling in downstream dataloaders.

4. **Intensity Normalization** — HU values are clipped to [−997, 888] and z-score normalized (μ = −142, σ = 361), derived from the 0.5% and 99.5% percentiles of foreground voxels across a random sample of 1,000 CT-RATE scans.

5. **Multi-Crop Strategy** — Two global crops (size 256) from the center slice and eight local crops (size 144) sampled throughout the 12 mm window, with 80% probability of centering on RAD-ChestCT labels (falling back to TotalSegmentator masks when absent).

---

## Repository Structure

```
.
├── README.md                          # This file
│
├── 1_preprocessing/                   # Stage 1: Raw NIfTI → NPY conversion
│   ├── preprocess_val.py              # Validation set: NIfTI → NPY (HU + transpose)
│   ├── preprocess_val_resized.py      # Validation set: NIfTI → NPY (HU + 1mm isotropic resample)
│   └── preprocess_rad.py              # RAD-ChestCT: NPZ → NPY (1mm isotropic resample)
│
├── 2_label_generation/                # Stage 2: Label creation & metadata extraction
│   ├── create_rad_labels.py           # Collapse RAD-ChestCT location columns → 16 evaluation classes
│   └── generate_totalseg_metadata.py  # Per-slice TS class presence matrix (CSV)
│
├── 3_shard_creation/                  # Stage 3: NPY + masks → WebDataset shards
│   ├── create_shards.py               # v1: Basic CT-only shards (no masks)
│   ├── create_final_shards.py         # v2: CT + TS + ReX masks (single-stream)
│   ├── create_final_shards_2.py       # v3: CT + TS + ReX (dual-stream: rex-shards + standard shards)
│   ├── create_final_shards_3.py       # v4: v3 + TorchIO 1mm isotropic resampling
│   ├── create_final_shards_multi_res.py # v5: Multi-resolution multiplexed shards
│   ├── create_shards_subset.py        # Extract balanced subset from existing shards
│   └── add_ts_and_rex_to_shards.py    # Retrofit TS + ReX masks onto existing shards
│
├── 4_auditing/                        # Stage 4: Statistical analysis & quality control
│   ├── audit_masks.py                 # Global class frequencies + pos_weight recommendations
│   ├── calculate_voxel_stats.py       # Foreground HU statistics (mean, std, percentiles)
│   ├── check_orientations.py          # DICOM orientation distribution report
│   ├── check_resolution_dist.py       # Spatial resolution & Z-spacing histograms
│   ├── inspect_slice_outliers.py      # Identify anomalous scans (>500 slices)
│   └── rad_slice_count_histogram.py   # RAD-ChestCT slice count distribution
│
├── 5_inspection/                      # Stage 5: Shard visualization & debugging
│   ├── inspect_shards.py              # Visualize standard shards (CT + TS + ReX overlay)
│   ├── inspect_multires_shards.py     # Visualize multi-resolution multiplexed shards
│   ├── test_single_shard.py           # Debug single shard with ReX data
│   └── test_correction.py             # Verify HU correction integrity
│
└── inspected_samples/                 # Output directory for visualization PNGs (gitignored)
```

---

## Pipeline Stages

### Stage 1: Preprocessing (`1_preprocessing/`)

Convert raw NIfTI/NPZ volumes into optimized NumPy arrays.

| Script | Input | Output | Key Features |
|--------|-------|--------|-------------|
| [`preprocess_val.py`](1_preprocessing/preprocess_val.py) | `.nii.gz` | `.npy` (D,H,W float16) | HU conversion, transpose, blocklist filtering |
| [`preprocess_val_resized.py`](1_preprocessing/preprocess_val_resized.py) | `.nii.gz` | `.npy` (D,H,W float16) | HU conversion + TorchIO 1mm isotropic B-spline resampling, highest-resolution-per-group selection |
| [`preprocess_rad.py`](1_preprocessing/preprocess_rad.py) | `.npz` | `.npy` (D,H,W float16) | RAD-ChestCT NPZ → 1mm isotropic resampling via TorchIO |

**Usage Example:**
```bash
python 1_preprocessing/preprocess_val_resized.py \
    --data_dir /data/ct_rate/valid \
    --metadata /data/ct_rate/valid_metadata.csv \
    --output_dir /data/ct_rate/valid_npy \
    --blocklist /data/no_chest_valid.txt \
    --num_workers 16
```

### Stage 2: Label Generation (`2_label_generation/`)

Create structured label files and per-slice anatomical metadata.

| Script | Input | Output | Key Features |
|--------|-------|--------|-------------|
| [`create_rad_labels.py`](2_label_generation/create_rad_labels.py) | RAD-ChestCT split CSVs | `rad_labels.csv` | Collapses 100+ location-specific columns into 16 CT-RATE evaluation classes (Calcification, Cardiomegaly, etc.) |
| [`generate_totalseg_metadata.py`](2_label_generation/generate_totalseg_metadata.py) | `.npy` volumes + TS masks | Per-volume CSV | Binary presence matrix (slice × 117 classes) using memory-mapped reads |

**Usage Example:**
```bash
python 2_label_generation/generate_totalseg_metadata.py \
    --dataset_dir /data/ct_rate/train_npy \
    --mask_base_dir /data/totalsegmentator \
    --output_dir /data/ts_metadata \
    --num_workers 8
```

### Stage 3: Shard Creation (`3_shard_creation/`)

The core dataset compilation stage. Converts preprocessed NPY volumes and masks into streamable WebDataset `.tar` shards.

#### Version Evolution

| Version | Script | Key Innovation |
|---------|--------|---------------|
| v1 | [`create_shards.py`](3_shard_creation/create_shards.py) | Basic CT-only shards with labels, no masks |
| v2 | [`create_final_shards.py`](3_shard_creation/create_final_shards.py) | Adds TS + ReX masks, single-stream output |
| v3 | [`create_final_shards_2.py`](3_shard_creation/create_final_shards_2.py) | **Dual-stream routing**: ReX-positive scans → `rex-shards-*.tar`, others → `shards-*.tar`. Stacked PNG encoding for overlapping ReX findings |
| v4 | [`create_final_shards_3.py`](3_shard_creation/create_final_shards_3.py) | v3 + TorchIO 1mm isotropic resampling (B-spline for CT, nearest-neighbor for masks) |
| v5 | [`create_final_shards_multi_res.py`](3_shard_creation/create_final_shards_multi_res.py) | **Multi-resolution multiplexing**: Groups multiple reconstructions of the same scan, aligns slices by physical Z-depth, stores all resolutions in one sample |

**Shard Output Format (v3/v4):**
Each sample in a `.tar` shard contains:
```python
{
    "__key__": "train_1_a_1_0050",
    "npy": <float16 2D CT slice>,
    "mask.png": <PNG-encoded TotalSegmentator mask (128×128)>,
    "rex_mask.png": <PNG-encoded stacked ReX binary masks (128×128 × N classes)>,
    "json": {
        "original_file": "train_1_a_1.nii.gz",
        "slice_index": 50,
        "dataset_split": "train",
        "transform_slope": 1.0,
        "transform_inter": -1024.0,
        "z_spacing": 1.0,
        "original_shape": [512, 512, 300],
        "labels": [0, 1, 0, ...],
        "class_names": ["Class1", "Class2", ...],
        "rex_active_classes": ["1a", "2c"]
    }
}
```

**Dual-Stream Routing (v3+):**
- Scans with ReX findings → `rex-shards-w{worker:02d}-{idx:06d}.tar` (half shard size to account for stacked PNGs)
- Scans without ReX findings → `shards-w{worker:02d}-{idx:06d}.tar`
- Enables targeted sampling and class-balancing strategies during training

**Multi-Resolution Multiplexing (v5):**
- Groups multiple reconstructions of the same patient/study by `base_scan_id`
- Aligns slices across reconstructions by physical Z-depth (mm) using nearest-neighbor matching
- Each sample stores all reconstructions with per-reconstruction metadata

**Usage Example:**
```bash
python 3_shard_creation/create_final_shards_3.py \
    --data_dir /data/ct_rate/train \
    --metadata /data/ct_rate/train_metadata.csv \
    --labels /data/ct_rate/train_predicted_labels.csv \
    --ts_dir /data/totalsegmentator \
    --rex_json /data/rex/dataset.json \
    --rex_dir /data/rex/segmentations \
    --output_dir /data/shards \
    --blocklist /data/no_chest_train.txt \
    --num_workers 16 \
    --shard_size 5000
```

**Retrofitting Masks:**
```bash
python 3_shard_creation/add_ts_and_rex_to_shards.py \
    --input_shards /data/shards \
    --output_dir /data/shards_with_masks \
    --ts_dir /data/totalsegmentator \
    --rex_json /data/rex/dataset.json \
    --rex_dir /data/rex/segmentations \
    --num_workers 16
```

**Subset Extraction:**
```bash
python 3_shard_creation/create_shards_subset.py \
    --input_shards /data/shards \
    --output_dir /data/subset \
    --target_slices 500000 \
    --num_workers 16
```

### Stage 4: Auditing & Statistics (`4_auditing/`)

Compute dataset-wide statistics for normalization, class balancing, and quality control.

| Script | Purpose |
|--------|---------|
| [`audit_masks.py`](4_auditing/audit_masks.py) | Computes global voxel counts, slice presence, and volumetric frequencies for all 118 TS classes + 14 ReX classes. Outputs recommended `pos_weight` arrays for BCEWithLogitsLoss. |
| [`calculate_voxel_stats.py`](4_auditing/calculate_voxel_stats.py) | Computes foreground HU statistics (mean, std, 0.05th/0.5th/99.5th percentiles) using morphological body masking. Supports both WebDataset and NPZ input modes. Generates diagnostic overlay PNGs. |
| [`check_orientations.py`](4_auditing/check_orientations.py) | Reports DICOM ImageOrientationPatient distribution across datasets. |
| [`check_resolution_dist.py`](4_auditing/check_resolution_dist.py) | Histograms of XY resolution, Z-spacing, and slice counts from metadata CSV. |
| [`inspect_slice_outliers.py`](4_auditing/inspect_slice_outliers.py) | Identifies scans with >500 slices and cross-references metadata to find root causes (thin Z-spacing, specific series descriptions). |
| [`rad_slice_count_histogram.py`](4_auditing/rad_slice_count_histogram.py) | Terminal histogram of RAD-ChestCT slice counts in bins of 50. |

**Usage Example:**
```bash
# Compute foreground HU statistics
python 4_auditing/calculate_voxel_stats.py \
    --input_mode wds \
    --shards_path "/data/shards/*.tar" \
    --max_samples 5000 \
    --stride 50 \
    --num_workers 8 \
    --output_dir /outputs/mask_checks

# Audit class frequencies
python 4_auditing/audit_masks.py \
    --ts_dir /data/totalsegmentator \
    --rex_json /data/rex/dataset.json \
    --rex_dir /data/rex/segmentations \
    --num_workers 16
```

### Stage 5: Inspection & Debugging (`5_inspection/`)

Visualize shard contents and verify data integrity.

| Script | Purpose |
|--------|---------|
| [`inspect_shards.py`](5_inspection/inspect_shards.py) | Visualizes standard shards: CT slice + TS mask overlay + ReX mask overlay. Prioritizes `rex-shards-*.tar`. |
| [`inspect_multires_shards.py`](5_inspection/inspect_multires_shards.py) | Visualizes multi-resolution multiplexed shards: N×3 grid (CT, TS, ReX) for each reconstruction at a physical Z-depth. |
| [`test_single_shard.py`](5_inspection/test_single_shard.py) | Debug utility: processes a single shard containing ReX data to verify 4D→2D stacking logic. |
| [`test_correction.py`](5_inspection/test_correction.py) | Verifies that manual HU correction (slope/intercept application) matches nibabel's automatic correction. |

**Usage Example:**
```bash
python 5_inspection/inspect_shards.py /data/shards --num_shards 10
python 5_inspection/inspect_multires_shards.py /data/multi_res_shards --num_shards 5
```

---

## Normalization Parameters

The following parameters were empirically derived from the 0.5% and 99.5% percentiles of foreground voxels across a random sample of 1,000 CT-RATE scans (see [`calculate_voxel_stats.py`](4_auditing/calculate_voxel_stats.py)):

| Parameter | Value |
|-----------|-------|
| HU Clip Range | [−997, 888] |
| Mean (μ) | −142 |
| Std (σ) | 361 |

These are applied during training (not in this preprocessing pipeline) to z-score normalize each input slice.

---

## Annotation Classes

### TotalSegmentator (118 classes)

Covers organs, bones, muscles, and vessels. Class indices 1–117 (0 = background). Full class mapping is defined in [`generate_totalseg_metadata.py`](2_label_generation/generate_totalseg_metadata.py).

### ReXGroundingCT (14 radiological findings)

| Code | Finding |
|------|---------|
| 1a | Calcification |
| 1b | Cardiomegaly |
| 1c | Pericardial effusion |
| 1d | Hiatal hernia |
| 1e | Emphysema |
| 1f | Atelectasis |
| 2a | Lung nodule |
| 2b | Lung opacity |
| 2c | Pulmonary fibrotic sequela |
| 2d | Pleural effusion |
| 2e | Peribronchial thickening |
| 2f | Consolidation |
| 2g | Bronchiectasis |
| 2h | Interlobular septal thickening |

Additional classes tracked in RAD-ChestCT labels (see [`create_rad_labels.py`](2_label_generation/create_rad_labels.py)):
- Lymphadenopathy
- Medical material

---

## Dependencies

```
numpy>=1.21
scipy>=1.7
pandas>=1.3
nibabel>=3.2
webdataset>=0.2
opencv-python>=4.5
matplotlib>=3.4
tqdm>=4.62
torchio>=0.18
torch>=1.10
```

Install with:
```bash
pip install numpy scipy pandas nibabel webdataset opencv-python matplotlib tqdm torchio torch
```

---

## Multi-Crop Strategy (Training-Time)

The multi-crop strategy described in the paper is implemented at training time (not in this preprocessing pipeline). The key parameters are:

| Parameter | Value |
|-----------|-------|
| Slab thickness | 12 mm (dynamically computed from Z-spacing) |
| Global crops | 2 × 256×256 (center slice only) |
| Local crops | 8 × 144×144 (throughout slab) |
| Label-centering probability | 80% (RAD-ChestCT → TotalSegmentator fallback) |

---

## Citation

If you use this preprocessing pipeline in your research, please cite the corresponding paper.

---

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
