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
from scipy.io import savemat
from typing import Optional

from utils import *

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats" # brats2021
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/IP_Adapter/brats_segmentation_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.7
THRESHOLD_RATIO = 0.3
VALID_MODALITIES = ["t1", "t1ce", "t2", "flair"]

INSTRUCTION_TEMPLATES = [
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
    "Provide a tumor segmentation for this {MODALITY} brain MRI slice."
]

SLICES_PER_VOLUME = 8
LOW_RATIO = 0.3
HIGH_RATIO = 0.7

# SDF encoder instance
mask_encoder = MedicalSDFEncoder(normalize_per_instance=True)

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
    pat = re.compile(r"BraTS2021_(\d+)_") # matches BraTS2021_<digits>_
    uids: set[str] = set()
    
    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        
        if m:
            uids.add(m.group(1))
    
    return sorted(uids, key=int)

# Get all modality file paths for a given subject ID
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
            # print(f"[WARNING] {case_id} {m} modality not found.")
            continue
        
        files[m] = matches[0]
    
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

def make_instruction(modality: str, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    
    return template.format(MODALITY=modality.upper())

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
    case_id = _WORKER_UIDS[idx]
    case_files = get_brats_case(case_id, BRATS_DIR)
    modalities = [k for k in case_files.keys() if k != "seg"]
    
    seg_volume, _ = load_nifti_image(case_files["seg"])
    label = (seg_volume > 0.5).astype(np.float32) # C, H, W binary mask
    
    for modality in modalities:
        try:
            img_volume, img_header_text = load_nifti_image(case_files[modality])
            image = img_volume.astype(np.float32)
            
            slice_indices = choose_slice_indices(label, SLICES_PER_VOLUME, PREFER_TUMOR_SLICE_PROB, rng=rng, low_ratio=LOW_RATIO, high_ratio=HIGH_RATIO)
            
            if not slice_indices:
                continue
            
            # Get mean and std of image volume
            image_mean, image_std = get_mean_and_std(image)
            
            # # Normalize image volume
            # image, _, _ = percentile_normalization(
            #     img_volume=image,
            #     modality=modality,
            #     use_mask=True, # use mask for normalization
            # )
            
            # # Make it to uint8 for saving
            # image = np.round(np.clip(image, 0, 1) * 255.0).astype(np.uint8)
            # label = np.round(np.clip(label, 0, 1) * 255.0).astype(np.uint8)
            
            # # Resize and pad
            # image = resize_and_pad(image, fill_value=0).astype(np.uint8)
            # label = resize_and_pad(label, target_size=512, fill_value=0).astype(np.uint8)
            
            for slice_idx in slice_indices:
                image_slice = image[slice_idx]
                label_slice = label[slice_idx]
                label_slice = mask_encoder.mask_to_sdf(label_slice)  # encode mask to SDF
                
                image_slice = (image_slice - image_mean) / (image_std + 1e-8)  # normalize using mean and std
                
                instruction = make_instruction(modality, rng=rng)
                
                data = {
                    "image": image_slice,
                    "label": label_slice,
                    "instruction": instruction, # string
                    "text": np.array("", dtype=object), # no extra text metadata available for brats dataset
                    "image_header": np.array(img_header_text, dtype=object)
                }
                
                out_path = _WORKER_OUT_DIR / f"brats_{case_id}_{modality}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                savemat(out_path, data)
        
        except Exception as e:
            # print(f"[ERROR] Failed to process {case_id} {modality}: {e}")
            continue

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(BRATS_DIR) / "seg")
    
    if not uids:
        raise RuntimeError(f"No BRATS cases found under {BRATS_DIR}")

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(len(uids))), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
            
    count = len(list(out_dir.glob("*.mat")))
    print(f"Saved {count} samples to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
