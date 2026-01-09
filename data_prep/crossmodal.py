import os
import random
import re
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import savemat

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats"
# OUTPUT_DIR = "./brats_crossmodal_mat"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat"
NUM_SAMPLES = 5000
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.7
VALID_MODALITIES = ["flair", "t1", "t1ce", "t2"]

INSTRUCTION_TEMPLATES = [
    "Translate this {src} MRI slice into {tgt}.",
    "Generate a {tgt} image from the {src} input.",
    "Synthesize {tgt} from {src} for this MRI slice.",
    "Convert this {src} brain MRI slice into {tgt}.",
    "Create the {tgt} modality from the {src} modality.",
    "Produce a {tgt} MRI slice given {src}.",
    "Map {src} to {tgt} for this slice.",
    "Given {src}, generate the corresponding {tgt} image.",
]


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


def make_instruction(src: str, tgt: str, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return template.format(src=src.upper(), tgt=tgt.upper())


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

    src, tgt = rng.sample(VALID_MODALITIES, 2)
    src_volume, header_text = load_nifti_data_and_header(case_files[src])
    tgt_volume = load_nifti_image(case_files[tgt])
    seg_volume = load_nifti_image(case_files["seg"])

    slice_idx = choose_slice_index(seg_volume, PREFER_TUMOR_SLICE_PROB, rng=rng)
    src_slice = src_volume[:, :, slice_idx].astype(np.float32)
    tgt_slice = tgt_volume[:, :, slice_idx].astype(np.float32)

    instruction = make_instruction(src, tgt, rng=rng)

    data = {
        "image": src_slice,
        "label": tgt_slice,
        "instruction": np.array(instruction, dtype=object),
        "text": np.array(header_text, dtype=object),
    }

    out_path = _WORKER_OUT_DIR / f"brats_{case_id}_{src}_to_{tgt}_slice_{slice_idx:03d}_{idx:06d}.mat"
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

    print(f"Saved {NUM_SAMPLES} samples to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
