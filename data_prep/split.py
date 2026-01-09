import os
import random
import shutil
from pathlib import Path


def split_dataset(
    src_dir,
    out_dir,
    ratios=(0.8, 0.1, 0.1),
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


if __name__ == "__main__":
    split_dataset(
        src_dir="/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat",
        out_dir="/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat",
        ratios=(0.8, 0.1, 0.1),
        seed=42,
        exts={".mat"},  # e.g. {'.jpg', '.png'}
        option="move",  # or "copy"
    )
