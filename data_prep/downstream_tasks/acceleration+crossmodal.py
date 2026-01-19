import sys
from pathlib import Path

# add parent directory to path for imports (so undersampling.py, utils.py are found)
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

import os
import random
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy.io import savemat

from undersampling import apply_fixed_mask


# -------------------------
# Configuration
# -------------------------
DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/brats_acceleration_crossmodal_mat"

NUM_SAMPLES = 5000
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 40)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.7
VALID_MODALITIES = ["flair", "t1", "t1ce", "t2"]

# Acceleration settings
ACS_NUM = 24
PARALLEL_FACTORS = [2, 4, 6, 8]  # R

# Instructions: "accelerate src, then convert to tgt"
INSTRUCTION_TEMPLATES = [
    "Accelerate this {SRC} MRI slice (R={R}) and convert it to {TGT}.",
    "Given this accelerated {SRC} slice (R={R}), generate the corresponding {TGT} image.",
    "Translate an accelerated (R={R}) {SRC} MRI slice into {TGT}.",
    "Convert this {SRC} slice to {TGT} under acceleration (R={R}).",
    "From an undersampled {SRC} slice (R={R}), reconstruct the clean {TGT} modality.",
    "Use this accelerated {SRC} MRI slice (R={R}) to produce a clean {TGT} slice.",
]


# -------------------------
# I/O helpers (BraTS)
# -------------------------
def load_nifti_image(file_path: Path) -> np.ndarray:
    img = nib.load(str(file_path))
    return img.get_fdata()


def load_nifti_data_and_header(file_path: Path) -> tuple[np.ndarray, str]:
    img = nib.load(str(file_path))
    return img.get_fdata(), img.header.__str__()


def find_all_uids(root: str | Path) -> list[str]:
    root = Path(root)
    uids = set()
    for p in root.glob("*.nii.gz"):
        m = re.search(r"BraTS2021_(\d+)_", p.name)
        if m:
            uids.add(m.group(1))
    return sorted(uids)


def get_brats_case(case_id: str, root: str | Path) -> dict[str, Path]:
    root = Path(root)
    seg_root = root / "seg"
    modality_root = root / "image"

    files: dict[str, Path] = {}
    seg_pattern = f"*_{case_id}_seg.nii.gz"
    seg_matches = list(seg_root.glob(seg_pattern))
    if len(seg_matches) != 1:
        raise FileNotFoundError(f"{case_id} seg: {seg_matches}")
    files["seg"] = seg_matches[0]

    for m in VALID_MODALITIES:
        pattern = f"*_{case_id}_{m}.nii.gz"
        matches = list(modality_root.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"{case_id} {m}: {matches}")
        files[m] = matches[0]

    return files


def choose_slice_index(seg_volume: np.ndarray, prefer_tumor_prob: float, rng: random.Random = random) -> int:
    depth = seg_volume.shape[2]
    tumor_slices = np.where(np.any(seg_volume != 0, axis=(0, 1)))[0]
    if tumor_slices.size > 0 and rng.random() < prefer_tumor_prob:
        return int(rng.choice(tumor_slices))
    return rng.randrange(depth)


def make_instruction(src: str, tgt: str, R: int, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return template.format(SRC=src.upper(), TGT=tgt.upper(), R=R)


# -------------------------
# Acceleration helper
# -------------------------
def apply_acceleration_to_slice(clean_slice_2d: np.ndarray, R: int) -> np.ndarray:
    """
    clean_slice_2d: (H, W) float
    returns: accelerated_slice_2d: (H, W) float with aliasing/undersampling artifacts
    """
    # apply_fixed_mask expects (C,H,W) in your fastmri pipeline; wrap 2D -> 3D with C=1.
    x = clean_slice_2d[None, ...].astype(np.float32)  # (1,H,W)

    y, _ = apply_fixed_mask(
        x,
        acs_num=ACS_NUM,
        parallel_factor=R,
    )
    y = np.abs(y).astype(np.float32)  # (1,H,W)
    return y[0]


# -------------------------
# Multiprocessing
# -------------------------
_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIR: Path | None = None
_WORKER_SEED = 0


def _init_worker(uids: list[str], out_dir: str, seed: int) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIR, _WORKER_SEED
    _WORKER_UIDS = uids
    _WORKER_OUT_DIR = Path(out_dir)
    _WORKER_SEED = seed


def _generate_sample(idx: int) -> None:
    if _WORKER_OUT_DIR is None:
        raise RuntimeError("Worker not initialized")

    rng = random.Random(_WORKER_SEED + idx)
    case_id = rng.choice(_WORKER_UIDS)
    case_files = get_brats_case(case_id, BRATS_DIR)

    # pick src/tgt modalities
    src, tgt = rng.sample(VALID_MODALITIES, 2)

    # load src/tgt volumes + header text for "text"
    src_volume, src_header_text = load_nifti_data_and_header(case_files[src])
    tgt_volume, _tgt_header_text = load_nifti_data_and_header(case_files[tgt])
    seg_volume = load_nifti_image(case_files["seg"])

    # choose slice (prefer tumor slices)
    slice_idx = choose_slice_index(seg_volume, PREFER_TUMOR_SLICE_PROB, rng=rng)

    # label = clean original target slice
    tgt_clean_slice = tgt_volume[:, :, slice_idx].astype(np.float32)

    # image = accelerated source slice
    src_clean_slice = src_volume[:, :, slice_idx].astype(np.float32)
    R = rng.choice(PARALLEL_FACTORS)
    src_acc_slice = apply_acceleration_to_slice(src_clean_slice, R=R)

    instruction = make_instruction(src, tgt, R, rng=rng)

    data = {
        "image": src_acc_slice,                               # accelerated {src}
        "label": tgt_clean_slice,                             # clean {tgt}
        "instruction": np.array(instruction, dtype=object),   # string
        "text": np.array(src_header_text, dtype=object),      # header text (string)
    }

    out_path = _WORKER_OUT_DIR / f"brats_{case_id}_{src}_R{R}_to_{tgt}_s{slice_idx:03d}_{idx:06d}.mat"
    savemat(out_path, data)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(BRATS_DIR) / "seg")
    if not uids:
        raise RuntimeError(f"No BRATS cases found under {BRATS_DIR}")

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(NUM_SAMPLES)), start=1):
            print(f"{done}/{NUM_SAMPLES} samples created", end="\r")

    count = len(list(out_dir.glob("*.mat")))
    print(f"\nSaved {count} samples to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
