import shutil
from pathlib import Path

SRC_DIRS = [
    "/fast_storage/intern/data/instruction_tuning/brats2023_acceleration_mat_t1tot1",
    "/fast_storage/intern/data/instruction_tuning/brats2023_crossmodal_mat_t1tot2",
    "/fast_storage/intern/data/instruction_tuning/brats2023_acceleration_mat_t1tot2",
    "/fast_storage/intern/data/instruction_tuning/brats2023_crossmodal_mat_t2toflair",
    "/fast_storage/intern/data/instruction_tuning/brats2023_crossmodal_mat_t1toflair",
]


def flatten_splits(src_dirs):
    """
    Move all files from train/val/test subfolders back to parent directory,
    then delete the subfolders.

    Args:
        src_dirs (list[str | Path]): list of directories with train/val/test subfolders
    """
    src_dirs = [Path(d) for d in src_dirs]
    
    for src_dir in src_dirs:
        print(f"\nProcessing {src_dir.name}...")
        
        # Iterate over train, val, test folders
        for split_name in ["train", "val", "test"]:
            split_dir = src_dir / split_name
            
            if not split_dir.exists():
                print(f"  {split_name} folder not found, skipping")
                continue
            
            # Move all files from split_dir to parent (src_dir)
            files = list(split_dir.glob("*.mat"))
            print(f"  Moving {len(files)} files from {split_name}/ to parent...")
            
            for f in files:
                dest = src_dir / f.name
                shutil.move(str(f), str(dest))
            
            # Remove the now-empty split folder
            split_dir.rmdir()
            print(f"  Deleted {split_name} folder")
    
    print("\n✓ Unsplit complete!")


if __name__ == "__main__":
    flatten_splits(SRC_DIRS)
