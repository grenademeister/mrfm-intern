import os
import random
import re
import shutil
from pathlib import Path


def split_dataset(
    src_dir,
    out_dir,
    ratios=(0.9, 0.05, 0.05),
    seed=42,
    exts=None,
    option="copy",
):
    """
    Split files in src_dir into train/val/test subfolders under out_dir.

    Args:
        src_dir (str | Path): source directory with files
        out_dir (str | Path): output directory
        ratios (tuple): (train, val, test) ratios, must sum to 1
        seed (int): random seed for reproducibility
        exts (set[str] | None): file extensions to include, e.g. {'.jpg', '.png'}
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    assert abs(sum(ratios) - 1.0) < 1e-6

    files = [p for p in src_dir.iterdir() if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]

    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    for split, split_files in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            if option == "copy":
                shutil.copy2(f, split_dir / f.name)
            elif option == "move":
                shutil.move(f, split_dir / f.name)

    print({k: len(v) for k, v in splits.items()})


def parse_filename(filename):
    """Extract case_id and slice_idx from filename.
    Examples:
    - fastmri_123_t1_r4_s001_n00001.mat -> (123, 1)
    - fastmri_456_t1_to_t2_s010_n00005.mat -> (456, 10)
    """
    match = re.search(r'fastmri_(\d+)_.*_s(\d+)_n\d+\.mat', filename)
    if match:
        case_id = match.group(1)
        slice_idx = int(match.group(2))
        return (case_id, slice_idx)
    return None


def split_paired_datasets(
    src_dirs,
    out_dirs,
    ratios=(0.8, 0.1, 0.1),
    max_samples=None,
    seed=42,
    exts=None,
    option="copy",
):
    """
    Split files from multiple directories with matching (case_id, slice_idx) pairs.
    Keeps only common pairs across all directories and splits them identically.

    Args:
        src_dirs (list[str | Path]): list of source directories
        out_dirs (list[str | Path]): list of output directories (must match src_dirs length)
        ratios (tuple): (train, val, test) ratios, must sum to 1
        max_samples (int | None): maximum number of samples to keep per directory
        seed (int): random seed for reproducibility
        exts (set[str] | None): file extensions to include, e.g. {'.mat'}
        option (str): "copy" or "move"
    """
    src_dirs = [Path(d) for d in src_dirs]
    out_dirs = [Path(d) for d in out_dirs]
    assert len(src_dirs) == len(out_dirs), "src_dirs and out_dirs must have same length"
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    # Collect files by (case_id, slice_idx) key for each directory
    files_by_key_per_dir = []
    for src_dir in src_dirs:
        files_by_key = {}
        all_files = [p for p in src_dir.iterdir() if p.is_file()]
        if exts is not None:
            all_files = [p for p in all_files if p.suffix.lower() in exts]
        
        for fpath in all_files:
            key = parse_filename(fpath.name)
            if key:
                if key not in files_by_key:
                    files_by_key[key] = []
                files_by_key[key].append(fpath)
        
        files_by_key_per_dir.append(files_by_key)
        print(f"{src_dir.name}: {len(files_by_key)} unique (case_id, slice) pairs")

    # Find common keys across all directories
    if not files_by_key_per_dir:
        print("No files found in any directory")
        return

    common_keys = set(files_by_key_per_dir[0].keys())
    for files_by_key in files_by_key_per_dir[1:]:
        common_keys = common_keys.intersection(set(files_by_key.keys()))
    
    common_keys = sorted(list(common_keys))
    print(f"\nFound {len(common_keys)} common (case_id, slice) pairs across all directories")

    # Limit to max_samples if specified
    if max_samples is not None and len(common_keys) > max_samples:
        random.seed(seed)
        common_keys = random.sample(common_keys, max_samples)
        print(f"Limited to {max_samples} samples")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(common_keys)

    n = len(common_keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_keys = {
        "train": common_keys[:n_train],
        "val": common_keys[n_train : n_train + n_val],
        "test": common_keys[n_train + n_val :],
    }

    # Copy/move files to split directories
    for dir_idx, (src_dir, out_dir) in enumerate(zip(src_dirs, out_dirs)):
        files_by_key = files_by_key_per_dir[dir_idx]
        
        for split, keys in split_keys.items():
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            file_count = 0
            for key in keys:
                if key in files_by_key:
                    for fpath in files_by_key[key]:
                        if option == "copy":
                            shutil.copy2(fpath, split_dir / fpath.name)
                        elif option == "move":
                            shutil.move(str(fpath), split_dir / fpath.name)
                        file_count += 1
            
            print(f"{src_dir.name} -> {split}: {file_count} files")
    
    # Print summary
    print("\nSplit summary:")
    for split, keys in split_keys.items():
        print(f"  {split}: {len(keys)} pairs")


if __name__ == "__main__":
    # Single directory split example
    # split_dataset(
    #     src_dir="/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat_new",
    #     out_dir="/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat_new",
    #     ratios=(0.9, 0.05, 0.05),
    #     seed=42,
    #     exts={".mat"},
    #     option="move",
    # )
    
    # Paired directories split example
    split_paired_datasets(
        src_dirs=[
            "/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat_t1",
            "/fast_storage/intern/data/instruction_tuning/fastmri_crossmodal_mat_t1tot2",
        ],
        out_dirs=[
            "/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat_t1",
            "/fast_storage/intern/data/instruction_tuning/fastmri_crossmodal_mat_t1tot2",
        ],
        ratios=(0.9, 0.05, 0.05),
        max_samples=10000,  # Set to None to keep all
        seed=42,
        exts={".mat"},
        option="move",  # Use "copy" for testing
    )
