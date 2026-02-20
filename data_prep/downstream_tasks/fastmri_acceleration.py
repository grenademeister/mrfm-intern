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

from undersampling import apply_fixed_mask
from utils import *

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
FASTMRI_DIR = f"{DATA_DIR}/fastmri"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

VALID_MODALITIES = ["t1", "t1ce", "t2", "flair"]

INSTRUCTION_TEMPLATES = [
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
    "Make this accelerated {MODALITY} brain MRI slice clean and faithful (R={R})."
]

# acceleration settings
ACS_NUM = 24
PARALLEL_FACTORS = [2, 4, 6, 8]

SLICES_PER_VOLUME = 8
SLICE_NUM_RATIO = 0.8

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

# Sample instructions for given acceleration factor R
def make_instruction(modality: str, R:int, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)

    return template.format(MODALITY=modality.upper(), R=R)

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
            
            # sample parallel factor
            pf = rng.choice(PARALLEL_FACTORS)
            
            # undersampled image
            image, _ = apply_fixed_mask(
                label,
                acs_num=ACS_NUM,
                parallel_factor=pf,
            )
            image = np.abs(image).astype(np.float32)
            
            # sample slices for each modality
            slices_num = int(label.shape[0] * SLICE_NUM_RATIO) # use only a portion of slices
            slices = rng.sample(range(slices_num), k=SLICES_PER_VOLUME)
            slices = sorted(slices)
            
            # get mean and std for normalization
            image_mean, image_std = get_mean_and_std(image)
            label_mean, label_std = get_mean_and_std(label)
            
            # # Normalize image and label volumes
            # image, _, _ = percentile_normalization(
            #     img_volume=image,
            #     modality=modality,
            #     use_mask=True,
            # )
            # label, _, _ = percentile_normalization(
            #     img_volume=label,
            #     modality=modality,
            #     use_mask=True,
            # )
            
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
                # Save as uint8
                image_slice = image[slice_idx]
                label_slice = label[slice_idx]
                
                
                # normalize slice
                image_slice = (image_slice - image_mean) / (image_std + 1e-8)
                label_slice = (label_slice - label_mean) / (label_std + 1e-8)
                
                instruction = make_instruction(modality, pf, rng=rng)

                data = {
                    "image": image_slice,
                    "label": label_slice,
                    "instruction": instruction, # string
                    "text": np.array(json_text, dtype=object),
                    "image_header": np.array(img_header_text, dtype=object)
                }

                out_path = _WORKER_OUT_DIR / f"fastmri_{case_id}_{modality}_r{pf}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
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
    
    # use 70% of the data for this task
    uids = uids[: int(len(uids) * 0.7)]
    
    if not uids:
        raise RuntimeError(f"No FASTMRI cases found under {FASTMRI_DIR}")

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(len(uids))), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
            
    count = len(list(out_dir.glob("*.mat")))
    print(f"Saved {count} samples to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
