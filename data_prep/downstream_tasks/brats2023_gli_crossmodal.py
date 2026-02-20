import sys
from pathlib import Path

# add parent directory to path for imports
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

import os
import random
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch
from scipy.io import savemat
from typing import Optional
import json

from utils import *

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats2023_gli"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 20)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.5
VALID_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
TEMPLATE_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Specific modality pairs to generate (src, tgt)
# If None, generate all possible pairs
# Example: [("t1", "t2"), ("t1ce", "flair")] to generate only these pairs
MODALITY_PAIRS = [("t1", "t2"), ("t2", "flair"), ("t1", "flair")]

# Generate output directory mapping for each pair
OUTPUT_BASE_DIR = "/fast_storage/intern/data/instruction_tuning"
if MODALITY_PAIRS is None:
    OUTPUT_DIRS = {}
else:
    OUTPUT_DIRS = {
        (src, tgt): f"{OUTPUT_BASE_DIR}/brats2023_gli_crossmodal_mat_{src}to{tgt}"
        for src, tgt in MODALITY_PAIRS
    }

INSTRUCTION_TEMPLATES = [
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
    "Turn this {SRC} MRI slice into a {TGT} MRI slice."
]

SLICES_PER_VOLUME = 8
LOW_RATIO = 0.3
HIGH_RATIO = 0.7

def load_nifti_image(file_path: Path) -> np.ndarray:
    img = nib.load(str(file_path))
    img = nib.as_closest_canonical(img) # reorient to RAS
    data = np.asanyarray(img.dataobj)
    data = data.astype(np.float32)
    
    assert data.ndim == 4 or data.ndim == 3, f"Expected 4D or 3D data, but shape of data is {data.shape}."
    
    # multi-coil case
    # (H, W, C, T)
    if data.ndim == 4:
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1)) # magnitude-only RSS
    
    data = data.transpose(2, 0, 1) # C, H, W
    data = np.ascontiguousarray(np.rot90(data, k=1, axes=(1, 2))) # correct rotation
    
    return data, img.header.__str__()

# Find all subject IDs in the dataset
def find_all_uids(root: str | Path) -> list[str]:
    root = Path(root)
    pat = re.compile(r"BraTS-GLI-(\d+)-000-") # only consider the first scan
    uids: set[str] = set()
    
    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        
        if m:
            uids.add(m.group(1))
    
    return sorted(uids, key=int)

# Get all modality file paths for a given subject ID
# keys: seg, t1, t1ce, t2, flair
def get_brats_case(case_id: str, root: str | Path) -> dict[str, Path]:
    root = Path(root)
    seg_root = root / "seg"
    modality_root = root / "image"
        
    files: dict[str, Path] = {}
    seg_pattern = f"BraTS-GLI-{case_id}-000-seg.nii.gz"
    seg_matches = list(seg_root.glob(seg_pattern))
    
    if len(seg_matches) != 1:
        raise FileNotFoundError(f"{case_id} seg: {seg_matches}")
    
    files["seg"] = seg_matches[0]
    
    for m in VALID_MODALITIES:
        pattern = f"BraTS-GLI-{case_id}-000-{m}.nii.gz"
        matches = list(modality_root.glob(pattern))
        
        if len(matches) != 1:
            # print(f"[WARNING] {case_id} {m} modality not found.")
            continue
        
        idx = VALID_MODALITIES.index(m)
        m_new = TEMPLATE_MODALITIES[idx] # map to template modality names
        files[m_new] = matches[0]
    
    return files

# Choose slice index with preference for tumor-containing slices
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
        # print("[WARNING] num_slices greater than volume depth.")
        return []
    
    # Determine valid slice range
    low = int(depth * low_ratio)
    high = int(depth * high_ratio)
    all_slices = list(range(low, high))
    tumor_slices = np.where(np.any(seg_volume != 0, axis=(1, 2)))[0].tolist() # slices with tumor
    
    # Remove excluded slices
    all_slices = [s for s in all_slices if s not in exclude_set]
    tumor_slices = [s for s in tumor_slices if s not in exclude_set]
    
    if num_slices > len(all_slices):
        # print("[WARNING] Not enough valid slices in the specified range.")
        return []
    
    chosen: set[int] = set()
    
    while len(chosen) < num_slices:
        if tumor_slices and rng.random() < prefer_tumor_prob:
            idx = rng.choice(tumor_slices)
        else:
            idx = rng.choice(all_slices)
        
        chosen.add(int(idx))
    
    return sorted(chosen)

def make_instruction(src_modality: str, tgt_modality: str, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    
    return template.format(SRC=src_modality.upper(), TGT=tgt_modality.upper())

_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIRS: dict[tuple[str, str], Path] = {}
_WORKER_SEED = 0

def _init_worker(uids: list[str], out_dirs: dict[tuple[str, str], str], seed: int) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIRS, _WORKER_SEED
    
    _WORKER_UIDS = uids
    _WORKER_OUT_DIRS = {pair: Path(dir_path) for pair, dir_path in out_dirs.items()}
    _WORKER_SEED = seed

def _generate_sample(idx: int) -> None:
    if not _WORKER_OUT_DIRS:
        raise RuntimeError("Worker not initialized")
    
    rng = random.Random(_WORKER_SEED + idx)
    case_id = _WORKER_UIDS[idx]
    case_files = get_brats_case(case_id, BRATS_DIR)
    modalities = [k for k in case_files.keys() if k != "seg"]
    
    seg_volume, _ = load_nifti_image(case_files["seg"])
    seg_volume = (seg_volume > 0.5).astype(np.uint8) # C, H, W binary mask
    
    # skip cases with less than 2 modalities
    if len(modalities) < 2:
        # print(f"[WARNING] {case_id} has less than 2 modalities, skipping.")
        return
    
    # Determine which modality pairs to process
    if MODALITY_PAIRS is None:
        # Generate all possible pairs randomly
        max_attempts = len(modalities)
        attempts = 0
        pairs_to_process = []
        
        temp_modalities = modalities.copy()
        while len(temp_modalities) >= 2 and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(range(len(temp_modalities)), 2)
            src, tgt = temp_modalities[i], temp_modalities[j]
            pairs_to_process.append((src, tgt))
            # remove used modalities to avoid duplicate pairs
            del temp_modalities[max(i, j)]
            del temp_modalities[min(i, j)]
    else:
        # Use only specified pairs that are available for this case
        pairs_to_process = [
            (src, tgt) for src, tgt in MODALITY_PAIRS
            if src in modalities and tgt in modalities
        ]
    
    if not pairs_to_process:
        return
    
    # Choose slice indices once for all pairs (for compositional generalization)
    slice_indices = choose_slice_indices(seg_volume, SLICES_PER_VOLUME, PREFER_TUMOR_SLICE_PROB, rng=rng, low_ratio=LOW_RATIO, high_ratio=HIGH_RATIO)
    
    if not slice_indices:
        return
    
    # Process each modality pair
    for src, tgt in pairs_to_process:
        
        try:
            img_volume, img_header_text = load_nifti_image(case_files[src])
            image = img_volume.astype(np.float32)
            
            label_volume, label_header_text = load_nifti_image(case_files[tgt])
            label = label_volume.astype(np.float32)
        except Exception as e:
            # print(f"[ERROR] Failed to load volumes for {case_id} {src} to {tgt}: {e}")
            continue
        
        # image_mean, image_std = get_mean_and_std(image)
        # label_mean, label_std = get_mean_and_std(label)
        
        # # Normalize image and label volumes
        # image, _, _ = percentile_normalization(
        #     img_volume=image,
        #     modality=src,
        #     use_mask=True,
        # )
        # label, _, _ = percentile_normalization(
        #     img_volume=label,
        #     modality=tgt,
        #     use_mask=True,
        # )
        
        # # Make it to uint8 for saving
        # image = np.round(np.clip(image, 0, 1) * 255.0).astype(np.uint8)
        # label = np.round(np.clip(label, 0, 1) * 255.0).astype(np.uint8)
        
        # # Resize and pad
        # image = resize_and_pad(image, fill_value=0).astype(np.uint8)
        # label = resize_and_pad(label, target_size=512, fill_value=0).astype(np.uint8)
        
        # src -> tgt
        for slice_idx in slice_indices:
            try:
                image_slice = image[slice_idx]
                label_slice = label[slice_idx]
                
                # Normalize using mean and std
                # image_slice = (image_slice - image_mean) / (image_std + 1e-8)
                # label_slice = (label_slice - label_mean) / (label_std + 1e-8)
                
                instruction = make_instruction(src, tgt, rng=rng)

                data = {
                    "image": image_slice,
                    "label": label_slice,
                    "instruction": np.array(instruction, dtype=object),
                    "text": np.array(label_header_text, dtype=object),
                    "image_header": np.array(img_header_text, dtype=object)
                }

                out_dir = _WORKER_OUT_DIRS[(src, tgt)]
                out_path = out_dir / f"brats2023_gli_{case_id}_{src}_to_{tgt}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                savemat(out_path, data)
            
            except Exception as e:
                # print(f"[ERROR] Failed to process slice {slice_idx} for {case_id} {src} to {tgt}: {e}")
                continue


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # Create all output directories
    for pair, out_dir_path in OUTPUT_DIRS.items():
        Path(out_dir_path).mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(BRATS_DIR) / "image")
    
    if not uids:
        raise RuntimeError(f"No BRATS cases found under {BRATS_DIR}")

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, OUTPUT_DIRS, SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(len(uids))), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
    
    print()
    # Print counts for each directory
    for pair, out_dir_path in OUTPUT_DIRS.items():
        count = len(list(Path(out_dir_path).glob("*.mat")))
        print(f"Saved {count} samples for {pair[0]}→{pair[1]} to {out_dir_path}")

if __name__ == "__main__":
    main()
