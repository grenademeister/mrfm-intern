import shutil
from pathlib import Path
import re
import random

SRC_DIRS = [
    "/fast_storage/intern/data/instruction_tuning/multi_task/acceleration",
    "/fast_storage/intern/data/instruction_tuning/multi_task/crossmodal",
    "/fast_storage/intern/data/instruction_tuning/multi_task/denoising",
    "/fast_storage/intern/data/instruction_tuning/multi_task/segmentation",
]
MAX_FILES = 20000  # Set to a number to limit files, or None for no limit
SEED = 42

# Split dataset into train/val/test folders
# Split across subjects to avoid data leakage
# Supports multiple source directories with synchronized splitting
def split_dataset(
    src_dirs,
    ratios=(0.95, 0.025, 0.025),
    option="move",
    max_files_per_dir=None,
    seed=42,
):
    """
    Split files in multiple src_dirs into train/val/test subfolders.
    All directories will use the same subject-based split to maintain consistency.

    Args:
        src_dirs (list[str | Path]): list of source directories with files
        ratios (tuple): (train, val, test) ratios, must sum to 1
        option (str): "copy" or "move"
        max_files_per_dir (int | None): if set, randomly select max_files per directory and delete the rest
        seed (int): random seed for reproducibility
    """
    random.seed(seed)
    src_dirs = [Path(d) for d in src_dirs]
    
    assert abs(sum(ratios) - 1.0) < 1e-6
    
    # Collect all files from all directories
    all_files_by_dir = {}
    for src_dir in src_dirs:
        files = list(src_dir.glob("*.mat"))
        all_files_by_dir[src_dir] = files
        print(f"Found {len(files)} files in {src_dir.name}")
    
    # Extract unique key from filenames
    # acceleration: brats2023_gli_00736_t1_to_t1_r4_s058_n00517.mat
    # crossmodal:   brats2023_gli_01005_t2_to_flair_s096_n00591.mat
    # Common key across dirs: (subject, slice) — r and n can differ between dirs
    pat = re.compile(r"_(?:gli|men)_(\d+).*_s(\d+)_n(\d+)\.mat$")
    
    subject_slice_by_dir = {}
    for src_dir, files in all_files_by_dir.items():
        tuples = set()
        for f in files:
            m = pat.search(f.name)
            if m:
                subj_id = int(m.group(1))
                slice_id = int(m.group(2))
                tuples.add((subj_id, slice_id))
        subject_slice_by_dir[src_dir] = tuples
    
    # Find common (subject, slice) tuples across all directories
    common_subjects = set.intersection(*subject_slice_by_dir.values())
    print(f"\nCommon (subject, slice) tuples across all directories: {len(common_subjects)}")
    
    if not common_subjects:
        raise ValueError("No common (subject, slice, n) tuples found across directories")
    
    # Sort and potentially limit the number of (subject, slice, n) tuples
    sorted_subjects = sorted(common_subjects)
    
    if max_files_per_dir is not None:
        # Calculate how many tuples we need
        target_tuples = max_files_per_dir
        if len(sorted_subjects) > target_tuples:
            print(f"Randomly selecting {target_tuples} (subject, slice) tuples per directory...")
            selected_subjects = set(random.sample(sorted_subjects, target_tuples))
            
            sorted_subjects = sorted(selected_subjects)
    
    # Determine split boundaries based on (subject, slice) tuples
    num_subjects = len(sorted_subjects)
    train_end_idx = int(num_subjects * ratios[0])
    val_end_idx = int(num_subjects * (ratios[0] + ratios[1]))
    
    train_subjects = set(sorted_subjects[:train_end_idx])
    val_subjects = set(sorted_subjects[train_end_idx:val_end_idx])
    test_subjects = set(sorted_subjects[val_end_idx:])
    
    print(f"\nSplit: {len(train_subjects)} train files, {len(val_subjects)} val files, {len(test_subjects)} test files")
    
    # Process each directory with the same split
    for src_dir in src_dirs:
        splits = {
            "train": [],
            "val": [],
            "test": [],
        }
        
        # Re-scan files (in case some were deleted)
        for file in src_dir.glob("*.mat"):
            m = pat.search(file.name)
            
            if m:
                subj_id = int(m.group(1))
                slice_id = int(m.group(2))
                tuple_key = (subj_id, slice_id)
                
                # Assign to split based on (subject, slice, n) tuple
                if tuple_key in train_subjects:
                    splits["train"].append(file)
                elif tuple_key in val_subjects:
                    splits["val"].append(file)
                elif tuple_key in test_subjects:
                    splits["test"].append(file)

        print(f"\n{src_dir.name}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test files")
        
        for split, split_files in splits.items():
            split_dir = src_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for f in split_files:
                if option == "copy":
                    shutil.copy2(f, split_dir / f.name)
                elif option == "move":
                    shutil.move(f, split_dir / f.name)


if __name__ == "__main__":
    split_dataset(
        src_dirs=SRC_DIRS,
        ratios=(0.95, 0.025, 0.025),
        option="move",
        max_files_per_dir=MAX_FILES,
        seed=SEED,
    )
