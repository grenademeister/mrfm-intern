import sys
from pathlib import Path

# add parent directory to path for imports
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

import os
import random
import re
from multiprocessing import Pool
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.io import savemat

from undersampling import apply_fixed_mask
from utils import *

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
OUTPUT_BASE_DIR = "/fast_storage/intern/data/instruction_tuning"

DATASETS = [
    {
        "name": "brats2023",
        "root": f"{DATA_DIR}/brats2023_men",
        "id_prefix": "BraTS-MEN",
        "case_prefix": "brats2023_men",
    },
    {
        "name": "brats2023",
        "root": f"{DATA_DIR}/brats2023_gli",
        "id_prefix": "BraTS-GLI",
        "case_prefix": "brats2023_gli",
    },
]

SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 15)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.5
SLICES_PER_VOLUME = 8
LOW_RATIO = 0.3
HIGH_RATIO = 0.7

# modalities
VALID_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
TEMPLATE_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# crossmodal pairs (src, tgt)
MODALITY_PAIRS = [("t1", "t2"), ("t2", "flair"), ("t1", "flair")]

# undersampling: generate (t1_undersampled -> t1) and (t1_undersampled -> t2)
ACCEL_MODALITY = "t1"
ACCEL_TARGETS = ["t1", "t2"]

# output directories
OUTPUT_DIRS_CROSS = {
    (src, tgt): f"{OUTPUT_BASE_DIR}/brats2023_crossmodal_mat_{src}to{tgt}"
    for src, tgt in MODALITY_PAIRS
}
OUTPUT_DIRS_ACCEL = {
    tgt: f"{OUTPUT_BASE_DIR}/brats2023_acceleration_mat_{ACCEL_MODALITY}to{tgt}"
    for tgt in ACCEL_TARGETS
}

INSTRUCTION_TEMPLATES_CROSS = [
    "Convert this {SRC} brain MRI slice to {TGT}.",
    "Translate this {SRC} MRI image into {TGT}.",
    "Generate a {TGT} MRI slice from this {SRC} slice.",
    "Synthesize a {TGT} brain MRI image from {SRC}.",
    "Transform this {SRC} MRI scan slice into {TGT}.",
    "Create a {TGT} version of this {SRC} brain MRI slice.",
    "Predict the {TGT} MRI appearance from this {SRC} image.",
    "Produce a {TGT} brain MRI slice given this {SRC} slice.",
    "Map this {SRC} brain MRI image to {TGT} contrast.",
    "Convert {SRC} to {TGT} for this brain MRI slice.",
    "Generate a {TGT} contrast image from this {SRC} MRI slice.",
    "Render this {SRC} brain MRI scan slice as {TGT}.",
    "Synthesize {TGT} contrast for this {SRC} MRI image.",
    "Create a {TGT} brain MRI slice using this {SRC} MRI slice.",
    "Turn this {SRC} MRI slice into a {TGT} MRI slice.",
]

INSTRUCTION_TEMPLATES_ACCEL = [
    "Dealias this {MODALITY} slice (R={R}).",
    "Reconstruct this {MODALITY} image (R={R}).",
    "Remove artifacts in this {MODALITY} MRI slice (R={R}).",
    "Fix aliasing in this {MODALITY} brain MRI slice (R={R}).",
    "Clean this {MODALITY} MRI reconstruction (R={R}).",
    "Denoise and dealias this {MODALITY} MRI image (R={R}).",
    "Reduce aliasing artifacts in this {MODALITY} brain MRI image (R={R}).",
    "Improve the quality of this {MODALITY} MRI slice (R={R}).",
    "Restore details in this {MODALITY} MRI scan (R={R}).",
    "Reconstruct a clean {MODALITY} brain MRI slice (R={R}).",
    "Dealias this undersampled {MODALITY} MRI slice (R={R}).",
    "Remove undersampling artifacts from this {MODALITY} MRI image (R={R}).",
    "Restore anatomy in this {MODALITY} brain MRI slice (R={R}).",
    "Reduce aliasing in this {MODALITY} MRI slice without over-smoothing (R={R}).",
    "Make this accelerated {MODALITY} brain MRI slice clean and faithful (R={R}).",
]

# acceleration settings
ACS_NUM = 24
PARALLEL_FACTORS = [2, 4, 6, 8]

def load_nifti_image(file_path: Path) -> np.ndarray:
    img = nib.load(str(file_path))
    img = nib.as_closest_canonical(img)  # reorient to RAS
    data = np.asanyarray(img.dataobj)
    data = data.astype(np.float32)

    assert data.ndim == 4 or data.ndim == 3, f"Expected 4D or 3D data, but shape of data is {data.shape}."

    # multi-coil case
    # (H, W, C, T)
    if data.ndim == 4:
        data = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1))  # magnitude-only RSS

    data = data.transpose(2, 0, 1)  # C, H, W
    data = np.ascontiguousarray(np.rot90(data, k=1, axes=(1, 2)))  # correct rotation

    return data, img.header.__str__()

def find_all_uids(root: str | Path, id_prefix: str) -> list[str]:
    root = Path(root)
    pat = re.compile(rf"{re.escape(id_prefix)}-(\d+)-000-")  # only consider the first scan
    uids: set[str] = set()

    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        if m:
            uids.add(m.group(1))

    return sorted(uids, key=int)

def get_brats_case(case_id: str, root: str | Path, id_prefix: str) -> dict[str, Path]:
    root = Path(root)
    seg_root = root / "seg"
    modality_root = root / "image"

    files: dict[str, Path] = {}
    seg_pattern = f"{id_prefix}-{case_id}-000-seg.nii.gz"
    seg_matches = list(seg_root.glob(seg_pattern))

    if len(seg_matches) != 1:
        raise FileNotFoundError(f"{case_id} seg: {seg_matches}")

    files["seg"] = seg_matches[0]

    for m in VALID_MODALITIES:
        pattern = f"{id_prefix}-{case_id}-000-{m}.nii.gz"
        matches = list(modality_root.glob(pattern))

        if len(matches) != 1:
            continue

        idx = VALID_MODALITIES.index(m)
        m_new = TEMPLATE_MODALITIES[idx]  # map to template modality names
        files[m_new] = matches[0]

    return files

def choose_slice_indices(
    seg_volume: np.ndarray,
    num_slices: int,
    prefer_tumor_prob: float,
    rng: Optional[random.Random] = None,
    low_ratio: float = LOW_RATIO,
    high_ratio: float = HIGH_RATIO,
    exclude: Optional[list[int]] = None,
) -> list[int]:
    rng = rng or random
    depth = seg_volume.shape[0]
    exclude_set = set(exclude) if exclude else set()

    if num_slices > depth:
        return []

    low = int(depth * low_ratio)
    high = int(depth * high_ratio)
    all_slices = list(range(low, high))
    tumor_slices = np.where(np.any(seg_volume != 0, axis=(1, 2)))[0].tolist()

    all_slices = [s for s in all_slices if s not in exclude_set]
    tumor_slices = [s for s in tumor_slices if s not in exclude_set]

    if num_slices > len(all_slices):
        return []

    chosen: set[int] = set()

    while len(chosen) < num_slices:
        if tumor_slices and rng.random() < prefer_tumor_prob:
            idx = rng.choice(tumor_slices)
        else:
            idx = rng.choice(all_slices)

        chosen.add(int(idx))

    return sorted(chosen)

def make_instruction_cross(src_modality: str, tgt_modality: str, rng: random.Random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES_CROSS)
    return template.format(SRC=src_modality.upper(), TGT=tgt_modality.upper())

def make_instruction_accel(modality: str, R: int, rng: random.Random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES_ACCEL)
    return template.format(MODALITY=modality.upper(), R=R)

_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIRS_CROSS: dict[tuple[str, str], Path] = {}
_WORKER_OUT_DIRS_ACCEL: dict[str, Path] = {}
_WORKER_SEED = 0
_WORKER_DATASET_ROOT = ""
_WORKER_ID_PREFIX = ""
_WORKER_CASE_PREFIX = ""

def _init_worker(
    uids: list[str],
    out_dirs_cross: dict[tuple[str, str], str],
    out_dirs_accel: dict[str, str],
    seed: int,
    dataset_root: str,
    id_prefix: str,
    case_prefix: str,
) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIRS_CROSS, _WORKER_OUT_DIRS_ACCEL
    global _WORKER_SEED, _WORKER_DATASET_ROOT, _WORKER_ID_PREFIX, _WORKER_CASE_PREFIX
    _WORKER_UIDS = uids
    _WORKER_OUT_DIRS_CROSS = {k: Path(v) for k, v in out_dirs_cross.items()}
    _WORKER_OUT_DIRS_ACCEL = {k: Path(v) for k, v in out_dirs_accel.items()}
    _WORKER_SEED = seed
    _WORKER_DATASET_ROOT = dataset_root
    _WORKER_ID_PREFIX = id_prefix
    _WORKER_CASE_PREFIX = case_prefix

def _generate_sample(idx: int) -> None:
    if not _WORKER_OUT_DIRS_CROSS or not _WORKER_OUT_DIRS_ACCEL:
        raise RuntimeError("Worker not initialized")

    rng = random.Random(_WORKER_SEED + idx)
    case_id = _WORKER_UIDS[idx]
    case_files = get_brats_case(case_id, _WORKER_DATASET_ROOT, _WORKER_ID_PREFIX)
    modalities = [k for k in case_files.keys() if k != "seg"]

    seg_volume, _ = load_nifti_image(case_files["seg"])
    seg_volume = (seg_volume > 0.5).astype(np.uint8)

    if len(modalities) < 2:
        return

    pairs_to_process = [
        (src, tgt) for src, tgt in MODALITY_PAIRS if src in modalities and tgt in modalities
    ]
    if not pairs_to_process:
        return

    if ACCEL_MODALITY not in modalities:
        return

    slice_indices = choose_slice_indices(
        seg_volume,
        SLICES_PER_VOLUME,
        PREFER_TUMOR_SLICE_PROB,
        rng=rng,
        low_ratio=LOW_RATIO,
        high_ratio=HIGH_RATIO,
    )
    if not slice_indices:
        return

    try:
        accel_src_volume, accel_src_header_text = load_nifti_image(case_files[ACCEL_MODALITY])
    except Exception:
        return

    try:
        pf = rng.choice(PARALLEL_FACTORS)
        accel_image, _ = apply_fixed_mask(
            accel_src_volume.astype(np.float32),
            acs_num=ACS_NUM,
            parallel_factor=pf,
        )
        accel_image = np.abs(accel_image).astype(np.float32)
    except Exception:
        return

    accel_tgt_volumes: dict[str, tuple[np.ndarray, str]] = {}
    for tgt in ACCEL_TARGETS:
        try:
            vol, header_text = load_nifti_image(case_files[tgt])
            accel_tgt_volumes[tgt] = (vol, header_text)
        except Exception:
            return

    for src, tgt in pairs_to_process:
        try:
            src_volume, src_header_text = load_nifti_image(case_files[src])
            tgt_volume, tgt_header_text = load_nifti_image(case_files[tgt])
        except Exception:
            continue

        for slice_idx in slice_indices:
            try:
                cross_instruction = make_instruction_cross(src, tgt, rng)
                cross_data = {
                    "image": src_volume[slice_idx],
                    "label": tgt_volume[slice_idx],
                    "instruction": np.array(cross_instruction, dtype=object),
                    "text": np.array(tgt_header_text, dtype=object),
                    "image_header": np.array(src_header_text, dtype=object),
                }
                cross_out_dir = _WORKER_OUT_DIRS_CROSS[(src, tgt)]
                cross_path = (
                    cross_out_dir
                    / f"{_WORKER_CASE_PREFIX}_{case_id}_{src}_to_{tgt}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                )
                savemat(cross_path, cross_data)

                for tgt_modality in ACCEL_TARGETS:
                    accel_tgt_volume, accel_tgt_header_text = accel_tgt_volumes[tgt_modality]
                    accel_instruction = make_instruction_accel(ACCEL_MODALITY, pf, rng)
                    accel_data = {
                        "image": accel_image[slice_idx],
                        "label": accel_tgt_volume[slice_idx],
                        "instruction": np.array(accel_instruction, dtype=object),
                        "text": np.array(accel_tgt_header_text, dtype=object),
                        "image_header": np.array(accel_src_header_text, dtype=object),
                    }
                    accel_out_dir = _WORKER_OUT_DIRS_ACCEL[tgt_modality]
                    accel_path = (
                        accel_out_dir
                        / f"{_WORKER_CASE_PREFIX}_{case_id}_{ACCEL_MODALITY}_to_{tgt_modality}_r{pf}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                    )
                    savemat(accel_path, accel_data)
            except Exception:
                continue

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    for out_dir in OUTPUT_DIRS_CROSS.values():
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    for out_dir in OUTPUT_DIRS_ACCEL.values():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        uids = find_all_uids(Path(ds["root"]) / "image", ds["id_prefix"])
        if not uids:
            raise RuntimeError(f"No BRATS cases found under {ds['root']}")

        with Pool(
            processes=WORKERS,
            initializer=_init_worker,
            initargs=(
                uids,
                OUTPUT_DIRS_CROSS,
                OUTPUT_DIRS_ACCEL,
                SEED,
                ds["root"],
                ds["id_prefix"],
                ds["case_prefix"],
            ),
        ) as pool:
            total = len(uids)
            for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(total)), start=1):
                pct = (done / total) * 100
                print(f"{ds['case_prefix']}: {done}/{total} ({pct:5.1f}%)", end="\r")
        print()

    for (src, tgt), out_dir in OUTPUT_DIRS_CROSS.items():
        count = len(list(Path(out_dir).glob("*.mat")))
        print(f"Saved {count} crossmodal samples for {src}->{tgt} to {out_dir}")
    for tgt, out_dir in OUTPUT_DIRS_ACCEL.items():
        count = len(list(Path(out_dir).glob("*.mat")))
        print(f"Saved {count} acceleration samples for {ACCEL_MODALITY}->{tgt} to {out_dir}")

if __name__ == "__main__":
    main()
