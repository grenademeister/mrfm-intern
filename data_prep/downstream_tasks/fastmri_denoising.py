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

from utils import *

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
FASTMRI_DIR = f"{DATA_DIR}/fastmri"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/fastmri_denoising_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

VALID_MODALITIES = ["t1", "t1ce", "t2", "flair"]

INSTRUCTION_TEMPLATES = [
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
    "Improve this {MODALITY} brain MRI slice by reducing {SEVERITY} noise."
]

DESCRIPTORS = {
    "low": ["subtle", "slight", "faint", "minimal", "mild"],
    "medium": ["moderate", "noticeable", "grainy", "visible", "prominent"],
    "high": ["heavy", "severe", "intense", "extreme", "coarse"]
}

SLICES_PER_VOLUME = 8
SLICE_NUM_RATIO = 0.8

NOISE_RANGE = (0.025, 0.15) # noise level range for denoising task
NOISE_LOW = 0.05
NOISE_HIGH = 0.125

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
    pat = re.compile(r"fastmri_(\d+)_") # matches fastmri_<digits>_
    uids: set[str] = set()
    
    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        
        if m:
            uids.add(m.group(1))

    return sorted(uids, key=int)

# Get all modality file paths for a given subject ID
def get_fastmri_case(case_id: str, root: str | Path) -> dict[str, Path]:
    root = Path(root)
    modality_root = root / "image"

    files: dict[str, Path] = {}
    pattern = f"*_{case_id}_*.nii.gz"

    for p in modality_root.glob(pattern):
        n = p.name.upper()
        
        # Check for corresponding JSON metadata file
        json_path = p.parent / (p.name.replace(".nii.gz", ".json"))

        if not json_path.exists():
            # print(f"[Warning] Metadata JSON file not found for case {case_id}: {p.name}.")
            continue
        
        if "T1" in n and ("POST" in n or "CONTRAST" in n):
            modality = "t1ce"
        elif "FLAIR" in n:
            modality = "flair"
        elif "T2" in n and ("POST" in n or "CONTRAST" in n): # skip T2 POST if exists
            # print(f"[WARNING] Skipping T2 POST modality for case {case_id}: {p.name}.")
            continue
        elif "T2" in n:
            modality = "t2"
        elif "T1" in n:
            modality = "t1"
        else:
            raise ValueError(f"Unrecognized modality: {n}")

        if modality in files:
            # print(f"[WARNING] Duplicate modality {modality} for case {case_id}: {files[modality].name}, {p.name}. Discarding the latter one.")
            continue

        files[modality] = p

    return files

# Generate instruction string
def make_instruction(modality: str, sigma: float, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    
    if sigma <= NOISE_LOW:
        severity = rng.choice(DESCRIPTORS["low"])
    elif sigma >= NOISE_HIGH:
        severity = rng.choice(DESCRIPTORS["high"])
    else:
        severity = rng.choice(DESCRIPTORS["medium"])

    return template.format(MODALITY=modality.upper(), SEVERITY=severity)

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
    case_files = get_fastmri_case(case_id, FASTMRI_DIR)

    for modality in list(case_files.keys()):
        try:
            img_volume, img_header_text = load_nifti_image(case_files[modality])
            label = img_volume.astype(np.float32) # C, H, W
            
            # sample slices for each modality
            slices_num = int(label.shape[0] * SLICE_NUM_RATIO) # use only a portion of slices
            slices = rng.sample(range(slices_num), k=SLICES_PER_VOLUME)
            slices = sorted(slices)
            
            # # normalize label to [0, 1]
            # label, _, _ = percentile_normalization(
            #     img_volume=label,
            #     modality=modality,
            #     use_mask=True,
            # )
            
            # get mean and std for normalization
            label_mean, label_std = get_mean_and_std(label)
            
            # normalize label volume
            label = (label - label_mean) / (label_std + 1e-8)
            
            # add Gaussian noise
            sigma = rng.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
            noise = np.random.normal(loc=0.0, scale=sigma, size=label.shape)
            image = label + noise
            
            # # Make it to uint8 for saving
            # image = np.round(np.clip(image, 0, 1) * 255.0).astype(np.uint8)
            # label = np.round(np.clip(label, 0, 1) * 255.0).astype(np.uint8)
            
            # # Resize and pad
            # image = resize_and_pad(image, fill_value=0).astype(np.uint8)
            # label = resize_and_pad(label, target_size=512, fill_value=0).astype(np.uint8)
            
            # Load metadata associated with the source image
            json_path = Path(str(case_files[modality]).replace(".nii.gz", ".json"))
            json_text = json_path.read_text()
            
            for slice_idx in slices:
                image_slice = image[slice_idx]
                label_slice = label[slice_idx]
                
                instruction = make_instruction(modality, sigma, rng=rng)

                data = {
                    "image": image_slice,
                    "label": label_slice,
                    "instruction": instruction, # string
                    "text": np.array(json_text, dtype=object),
                    "image_header": np.array(img_header_text, dtype=object)
                }

                out_path = _WORKER_OUT_DIR / f"fastmri_{case_id}_{modality}_sigma{sigma:.3f}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                savemat(out_path, data)
        
        except Exception as e:
                # print(f"Error in {case_id}/{modality}: {e}")
                continue

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(FASTMRI_DIR) / "image")
    
    # use last 30% of the data for this task
    uids = uids[int(len(uids) * 0.7):]
    
    if not uids:
        raise RuntimeError(f"No FASTMRI cases found under {FASTMRI_DIR}")

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(len(uids))), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
            
    count = len(list(out_dir.glob("*.mat")))
    print(f"Saved {count} samples to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
