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
OUTPUT_BASE_DIR = "/fast_storage/intern/data/instruction_tuning/multi_task"

DATASETS = [
    {
        "name": "brats2023_gli",
        "root": f"{DATA_DIR}/brats2023_gli",
        "id_prefix": "BraTS-GLI",
        "case_prefix": "brats2023_gli",
    },
    {
        "name": "brats2023_men",
        "root": f"{DATA_DIR}/brats2023_men",
        "id_prefix": "BraTS-MEN",
        "case_prefix": "brats2023_men",
    },
]

SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 20)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 1.0
SLICES_PER_VOLUME = 8
LOW_RATIO = 0.3
HIGH_RATIO = 0.7

# modalities
# VALID_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
# TEMPLATE_MODALITIES = ["t1", "t1ce", "t2", "flair"]

VALID_MODALITIES = ["t1n", "t2w"]
TEMPLATE_MODALITIES = ["t1", "t2"]

# crossmodal pairing
# "all": all ordered src->tgt pairs where src != tgt
# "random_nonoverlap": pair modalities randomly without reuse (like v2)
# "one_per_src": for each src, pick one random tgt (src != tgt)
# or a list/tuple of allowed pairs: [("t1", "t2"), ("t2", "t1")] -> choose one per subject
CROSSMODAL_PAIRING = "one_per_src"

# acceleration settings
ACS_NUM = 24
PARALLEL_FACTORS = [2, 4, 6, 8]

# denoising settings
NOISE_RANGE = (0.05, 0.2)
NOISE_LOW = 0.1
NOISE_HIGH = 0.15

INSTRUCTION_TEMPLATES_CROSS_NATURAL = [
    "Synthesize a {TGT} brain MRI slice from the provided {SRC} input data.",
    "Transform this {SRC} scan into a {TGT} image of the brain.",
    "Derive a synthetic {TGT} contrast from this {SRC}-weighted source image.",
    "Predict the {TGT} brain slice based on the given {SRC} MRI input.",
    "Generate a {TGT} magnetic resonance image from this {SRC} acquisition.",
    "Create a synthetic {TGT} brain MRI using the {SRC} input slice.",
    "Map this {SRC} data to a cross-modality {TGT} brain slice.",
    "Estimate a virtual {TGT} image of the brain from this {SRC} scan.",
    "Synthesize a representative {TGT}-weighted brain slice from {SRC} imaging.",
    "Produce a {TGT} MRI slice using the provided {SRC} input.",
    "Reconstruct a pseudo-{TGT} scan of the brain from {SRC} source information.",
    "Synthesize a {TGT} image from the corresponding {SRC} MRI slice.",
    "Translate this {SRC} modality into an axial {TGT} brain section.",
    "Compute a {TGT} contrast image based on the corresponding {SRC} scan.",
    "Generate a synthesized {TGT} representation of the brain from {SRC} input.",
]

INSTRUCTION_TEMPLATES_CROSS_CAPTION = [
    "Synthesized {TGT} brain MRI slice generated from {SRC} input data.",
    "{TGT} image of the brain transformed from a {SRC} scan.",
    "Synthetic {TGT} contrast derived from a {SRC}-weighted source image.",
    "Predicted {TGT} brain slice based on {SRC} MRI input.",
    "Generated {TGT} magnetic resonance image derived from {SRC} acquisition.",
    "Synthetic {TGT} brain MRI created from {SRC} input.",
    "Cross-modality synthesis showing a {TGT} brain slice mapped from {SRC} data.",
    "Virtual {TGT} image of the brain estimated from a {SRC} scan.",
    "Representative {TGT}-weighted brain slice synthesized from {SRC} imaging.",
    "Output {TGT} MRI slice generated from a {SRC} input.",
    "Pseudo-{TGT} scan of the brain reconstructed from {SRC} source information.",
    "{TGT} image synthesized from a corresponding {SRC} MRI slice.",
    "Axial {TGT} brain section produced via translation from {SRC} modality.",
    "Computed {TGT} contrast image based on the corresponding {SRC} scan.",
    "Synthesized {TGT} representation of the brain derived from {SRC} input.",
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

INSTRUCTION_TEMPLATES_DENOISE = [
    "Denoise this {MODALITY} brain MRI slice with {SEVERITY} noise.",
    "Remove noise from this {MODALITY} MRI image with {SEVERITY} noise.",
    "Clean this noisy {MODALITY} MRI slice with {SEVERITY} noise.",
    "Reduce noise in this {MODALITY} brain MRI image at {SEVERITY} level.",
    "Fix noise in this {MODALITY} MRI scan slice with {SEVERITY} noise.",
    "Denoise this {MODALITY} brain MRI image under {SEVERITY} noise.",
    "Suppress {SEVERITY} noise in this {MODALITY} MRI slice.",
    "Make this {MODALITY} brain MRI slice less noisy at {SEVERITY} level.",
    "Remove {SEVERITY} noise from this {MODALITY} brain MRI slice.",
    "Denoise this {MODALITY} MRI slice while keeping details under {SEVERITY} noise.",
    "Reduce {SEVERITY} noise in this {MODALITY} MRI scan without blur.",
    "Clean {SEVERITY} noise from this {MODALITY} brain MRI scan slice.",
    "Denoise this {MODALITY} brain MRI slice and preserve anatomy under {SEVERITY} noise.",
    "Remove {SEVERITY} noise from this {MODALITY} MRI image without smoothing.",
    "Improve this {MODALITY} brain MRI slice by reducing {SEVERITY} noise.",
]

INSTRUCTION_TEMPLATES_SEG = [
    "Segment the tumor in this {MODALITY} brain MRI slice.",
    "Identify and mask the tumor region in this {MODALITY} MRI image.",
    "Create a tumor segmentation mask for this {MODALITY} brain MRI slice.",
    "Locate the tumor area in this {MODALITY} MRI scan slice.",
    "Label tumor tissue in this {MODALITY} brain MRI image.",
    "Extract the tumor region from this {MODALITY} MRI slice.",
    "Mark the tumor location in this {MODALITY} brain MRI slice.",
    "Generate a tumor label map for this {MODALITY} MRI slice.",
    "Detect the tumor region in this {MODALITY} brain MRI image.",
    "Produce a tumor ROI mask from this {MODALITY} MRI scan slice.",
    "Separate tumor from normal brain in this {MODALITY} MRI slice.",
    "Annotate the tumor region in this {MODALITY} brain MRI image.",
    "Create a binary tumor map for this {MODALITY} brain MRI slice.",
    "Indicate the tumor region in this {MODALITY} MRI slice of the brain.",
    "Provide a tumor segmentation for this {MODALITY} brain MRI slice.",
]

DESCRIPTORS = {
    "low": ["subtle", "slight", "faint", "minimal", "mild"],
    "medium": ["moderate", "noticeable", "grainy", "visible", "prominent"],
    "high": ["heavy", "severe", "intense", "extreme", "coarse"],
}

# SDF encoder instance
mask_encoder = MedicalSDFEncoder(normalize_per_instance=True)

def load_nifti_image(file_path: Path) -> np.ndarray:
    img = nib.load(str(file_path))
    img = nib.as_closest_canonical(img)  # reorient to RAS
    data = np.asanyarray(img.dataobj)
    data = data.astype(np.float32)

    assert data.ndim == 4 or data.ndim == 3, f"Expected 4D or 3D data, but shape of data is {data.shape}."

    # multi-coil case
    if data.ndim == 4:
        data = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1))  # magnitude-only RSS

    data = data.transpose(2, 0, 1)  # C, H, W
    data = np.ascontiguousarray(np.rot90(data, k=1, axes=(1, 2)))  # correct rotation

    return data, img.header.__str__()

def find_all_uids(root: str | Path, id_prefix: str) -> list[str]:
    root = Path(root)
    pat = re.compile(rf"{re.escape(id_prefix)}-(\d+)-000-")
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
        m_new = TEMPLATE_MODALITIES[idx]
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

    vol_start = int(depth * low_ratio)
    vol_end = int(depth * high_ratio)
    global_valid_set = set(range(vol_start, vol_end)) - exclude_set

    raw_tumor_indices = np.where(np.any(seg_volume != 0, axis=(1, 2)))[0]
    raw_tumor_indices.sort()

    t_len = len(raw_tumor_indices)
    if t_len > 0:
        t_start_idx = int(t_len * LOW_RATIO)
        t_end_idx = int(t_len * HIGH_RATIO)
        if t_start_idx >= t_end_idx and t_len > 0:
            t_end_idx = min(t_start_idx + 1, t_len)
        restricted_tumor_indices = raw_tumor_indices[t_start_idx:t_end_idx]
    else:
        restricted_tumor_indices = []

    tumor_set = set(restricted_tumor_indices)
    all_tumor_set_original = set(raw_tumor_indices)
    non_tumor_set = global_valid_set - all_tumor_set_original

    valid_total_slices = len(tumor_set) + len(non_tumor_set)
    if num_slices > valid_total_slices:
        return []

    tumor_list = list(tumor_set)
    non_tumor_list = list(non_tumor_set)
    rng.shuffle(tumor_list)
    rng.shuffle(non_tumor_list)

    chosen = []
    while len(chosen) < num_slices:
        want_tumor = rng.random() < prefer_tumor_prob

        if (want_tumor and tumor_list) or not non_tumor_list:
            if tumor_list:
                chosen.append(tumor_list.pop())
            else:
                chosen = []
                break
        else:
            if non_tumor_list:
                chosen.append(non_tumor_list.pop())
            else:
                chosen = []
                break

    return sorted(chosen)

def make_instruction(templates: list[str], **kwargs) -> str:
    template = random.choice(templates)
    return template.format(**kwargs)

def make_instruction_denoise(modality: str, sigma: float, rng: random.Random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES_DENOISE)
    if sigma <= NOISE_LOW:
        severity = rng.choice(DESCRIPTORS["low"])
    elif sigma >= NOISE_HIGH:
        severity = rng.choice(DESCRIPTORS["high"])
    else:
        severity = rng.choice(DESCRIPTORS["medium"])
    return template.format(MODALITY=modality.upper(), SEVERITY=severity)

def get_crossmodal_pairs(modalities: list[str], rng: random.Random) -> list[tuple[str, str]]:
    if isinstance(CROSSMODAL_PAIRING, (list, tuple)):
        # Allow a fixed set of candidate pairs; pick one per subject.
        valid_pairs: list[tuple[str, str]] = []
        for item in CROSSMODAL_PAIRING:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            src, tgt = item
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            if src in modalities and tgt in modalities and src != tgt:
                valid_pairs.append((src, tgt))
        if not valid_pairs:
            return []
        return [rng.choice(valid_pairs)]
    if CROSSMODAL_PAIRING == "random_nonoverlap":
        pairs = []
        temp_modalities = modalities.copy()
        max_attempts = len(temp_modalities)
        attempts = 0
        while len(temp_modalities) >= 2 and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(range(len(temp_modalities)), 2)
            src, tgt = temp_modalities[i], temp_modalities[j]
            pairs.append((src, tgt))
            del temp_modalities[max(i, j)]
            del temp_modalities[min(i, j)]
        return pairs
    if CROSSMODAL_PAIRING == "one_per_src":
        pairs = []
        if len(modalities) < 2:
            return pairs
        for src in modalities:
            choices = [m for m in modalities if m != src]
            tgt = rng.choice(choices)
            pairs.append((src, tgt))
        return pairs

    # default: all ordered pairs
    return [(src, tgt) for src in modalities for tgt in modalities if src != tgt]

_WORKER_UIDS: list[str] = []
_WORKER_SEED = 0
_WORKER_DATASET_ROOT = ""
_WORKER_ID_PREFIX = ""
_WORKER_CASE_PREFIX = ""
_WORKER_OUT_DIRS: dict[str, Path] = {}

def _init_worker(
    uids: list[str],
    out_dirs: dict[str, str],
    seed: int,
    dataset_root: str,
    id_prefix: str,
    case_prefix: str,
) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIRS, _WORKER_SEED
    global _WORKER_DATASET_ROOT, _WORKER_ID_PREFIX, _WORKER_CASE_PREFIX
    _WORKER_UIDS = uids
    _WORKER_OUT_DIRS = {k: Path(v) for k, v in out_dirs.items()}
    _WORKER_SEED = seed
    _WORKER_DATASET_ROOT = dataset_root
    _WORKER_ID_PREFIX = id_prefix
    _WORKER_CASE_PREFIX = case_prefix

def _generate_sample(idx: int) -> None:
    if not _WORKER_OUT_DIRS:
        raise RuntimeError("Worker not initialized")

    rng = random.Random(_WORKER_SEED + idx)
    case_id = _WORKER_UIDS[idx]

    try:
        case_files = get_brats_case(case_id, _WORKER_DATASET_ROOT, _WORKER_ID_PREFIX)
    except Exception:
        return

    modalities = [k for k in case_files.keys() if k != "seg"]
    if len(modalities) < 2:
        return

    try:
        seg_volume, _ = load_nifti_image(case_files["seg"])
    except Exception:
        return

    seg_volume = (seg_volume > 0.5).astype(np.uint8)
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

    # load all modality volumes once
    modality_vols: dict[str, tuple[np.ndarray, str]] = {}
    for m in modalities:
        try:
            vol, header_text = load_nifti_image(case_files[m])
            modality_vols[m] = (vol.astype(np.float32), header_text)
        except Exception:
            return

    # --- Crossmodal ---
    pairs = get_crossmodal_pairs(modalities, rng)
    for src, tgt in pairs:
        try:
            src_volume, src_header_text = modality_vols[src]
            tgt_volume, tgt_header_text = modality_vols[tgt]

            for slice_idx in slice_indices:
                instruction = make_instruction(
                    INSTRUCTION_TEMPLATES_CROSS_NATURAL,
                    SRC=src.upper(),
                    TGT=tgt.upper(),
                )
                caption = make_instruction(
                    INSTRUCTION_TEMPLATES_CROSS_CAPTION,
                    SRC=src.upper(),
                    TGT=tgt.upper(),
                )

                data = {
                    "original": src_volume[slice_idx],
                    "image": src_volume[slice_idx],
                    "label": tgt_volume[slice_idx],
                    "instruction": np.array(instruction, dtype=object),
                    "caption": np.array(caption, dtype=object),
                    "text": np.array("", dtype=object),
                    "image_header": np.array(src_header_text, dtype=object),
                }

                out_path = (
                    _WORKER_OUT_DIRS["crossmodal"]
                    / f"{_WORKER_CASE_PREFIX}_{case_id}_{src}_to_{tgt}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                )
                savemat(out_path, data)
        except Exception:
            continue

    # --- Acceleration ---
    for modality in modalities:
        try:
            accel_src_volume, accel_src_header_text = modality_vols[modality]
            pf = rng.choice(PARALLEL_FACTORS)
            accel_image, _ = apply_fixed_mask(
                accel_src_volume.astype(np.float32),
                acs_num=ACS_NUM,
                parallel_factor=pf,
            )
            accel_image = np.abs(accel_image).astype(np.float32)
        except Exception:
            accel_image = None

        if accel_image is None:
            continue

        # target is always the original (same modality)
        tgt_volume, tgt_header_text = modality_vols[modality]
        for slice_idx in slice_indices:
            instruction = make_instruction(
                INSTRUCTION_TEMPLATES_ACCEL,
                MODALITY=modality.upper(),
                R=pf,
            )
            data = {
                "image": accel_image[slice_idx],
                "label": tgt_volume[slice_idx],
                "instruction": np.array(instruction, dtype=object),
                "text": np.array(tgt_header_text, dtype=object),
                "image_header": np.array(accel_src_header_text, dtype=object),
            }
            out_path = (
                _WORKER_OUT_DIRS["acceleration"]
                / f"{_WORKER_CASE_PREFIX}_{case_id}_{modality}_r{pf}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
            )
            savemat(out_path, data)

    # --- Denoising ---
    for modality in modalities:
        try:
            label = modality_vols[modality][0].astype(np.float32)
            label_mean, label_std = get_mean_and_std(label)
            label = (label - label_mean) / (label_std + 1e-8)
            sigma = rng.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
            noise = np.random.normal(loc=0.0, scale=sigma, size=label.shape)
            image = label + noise

            for slice_idx in slice_indices:
                instruction = make_instruction_denoise(modality, sigma, rng)
                data = {
                    "image": image[slice_idx],
                    "label": label[slice_idx],
                    "instruction": np.array(instruction, dtype=object),
                    "text": np.array("", dtype=object),
                    "image_header": np.array(modality_vols[modality][1], dtype=object),
                }
                out_path = (
                    _WORKER_OUT_DIRS["denoising"]
                    / f"{_WORKER_CASE_PREFIX}_{case_id}_{modality}_sigma{sigma:.3f}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                )
                savemat(out_path, data)
        except Exception as e:
            print("DENOISE ERROR:", case_id, modality, e)
            raise

    # --- Segmentation ---
    label_mask = (seg_volume > 0.5).astype(np.uint8)
    for modality in modalities:
        try:
            image = modality_vols[modality][0].astype(np.float32)

            for slice_idx in slice_indices:
                image_slice = image[slice_idx]
                label_slice = label_mask[slice_idx]
                instruction = make_instruction(INSTRUCTION_TEMPLATES_SEG, MODALITY=modality.upper())
                data = {
                    "image": image_slice,
                    "label": label_slice,
                    "instruction": np.array(instruction, dtype=object),
                    "text": np.array("", dtype=object),
                    "image_header": np.array(modality_vols[modality][1], dtype=object),
                }
                out_path = (
                    _WORKER_OUT_DIRS["segmentation"]
                    / f"{_WORKER_CASE_PREFIX}_{case_id}_{modality}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                )
                savemat(out_path, data)
        except Exception as e:
            print("SEG ERROR:", case_id, modality, e)
            raise

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dirs = {
        "crossmodal": f"{OUTPUT_BASE_DIR}/crossmodal",
        "acceleration": f"{OUTPUT_BASE_DIR}/acceleration",
        "denoising": f"{OUTPUT_BASE_DIR}/denoising",
        "segmentation": f"{OUTPUT_BASE_DIR}/segmentation",
    }

    for out_dir in out_dirs.values():
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
                out_dirs,
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

    for name, out_dir in out_dirs.items():
        count = len(list(Path(out_dir).glob("*.mat")))
        print(f"Saved {count} samples to {out_dir} ({name})")

if __name__ == "__main__":
    main()
