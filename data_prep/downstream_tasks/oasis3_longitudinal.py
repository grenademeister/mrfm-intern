import sys
from pathlib import Path

# add parent directory to path for imports
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

import os
import random
import re
import json

import nibabel as nib
import numpy as np
from itertools import combinations
from multiprocessing import Pool
from scipy.io import savemat

from utils import *

DATA_DIR = "/fast_storage/intern/data/data_curation"
OASIS3_DIR = f"{DATA_DIR}/oasis3"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 10)

INSTRUCTION_TEMPLATES = [
    "Predict this {MODALITY} brain MRI slice after {MONTHS} months.",
    "Forecast this {MODALITY} MRI image {MONTHS} months later.",
    "Generate the {MONTHS}-month follow-up {MODALITY} MRI slice.",
    "Simulate this {MODALITY} brain MRI scan slice at month {MONTHS}.",
    "Synthesize a {MODALITY} MRI slice representing {MONTHS} months later.",
    "Create a future {MODALITY} brain MRI slice for +{MONTHS} months.",
    "Predict the next-timepoint {MODALITY} brain MRI image at {MONTHS} months.",
    "Produce a {MODALITY} brain MRI slice showing changes after {MONTHS} months.",
    "Generate a {MODALITY} follow-up MRI image at {MONTHS} months from the current scan.",
    "Predict a {MODALITY} brain MRI slice at {MONTHS} months and keep anatomy consistent.",
    "Simulate the {MODALITY} brain MRI appearance {MONTHS} months after this input slice.",
    "Create a {MODALITY} MRI scan slice for the {MONTHS}-month follow-up timepoint.",
    "Forecast the {MODALITY} brain MRI slice after {MONTHS} months while preserving structures.",
    "Predict the {MONTHS}-month future {MODALITY} brain MRI image without introducing artifacts.",
    "Generate a plausible {MODALITY} brain MRI slice at {MONTHS} months that matches expected progression."
]

SLICES_PER_VOLUME = 8
LOW_RATIO = 0.3
HIGH_RATIO = 0.7
VOLUME_INTENSITY_THRESHOLD = 45.0
SLICE_INTENSITY_THRESHOLD = 45.0

def extract_days(path: str | Path) -> int:
    name = path.name if isinstance(path, Path) else path
    match = re.search(r"sess?-d(\d+)", name) # matches d0, d100, etc.
    
    return int(match.group(1)) if match else 0

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
    pat = re.compile(r"oasis3_sub-OAS(\d+)_") # matches oasis3_sub-OAS<digits>_
    uids: set[str] = set()
    
    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        
        if m:
            uids.add(m.group(1))

    return sorted(uids, key=int)

# Get all modality file paths for a given subject ID
def get_oasis3_case(case_id: str, root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    modality_root = root / "image"

    files: dict[str, list[Path]] = {}
    pattern = f"*OAS{case_id}_*.nii.gz"

    for p in modality_root.glob(pattern):
        n = p.name.upper()
        
        # Skip multi-echo sequences (keep only primary echoes)
        if "ECHO-" in n:
            # print(f"[WARNING] Skip multi-echo sequence {n}")
            continue
        
        # Check for corresponding JSON metadata file
        json_path = p.parent / (p.name.replace(".nii.gz", ".json"))

        if not json_path.exists():
            # print(f"[WARNING] JSON not found for {n}")
            continue
        
        # Validate metadata quality
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
                # Require at least 15 metadata fields for data quality
                if len(metadata) < 15:
                    # print(f"[WARNING] Insufficient metadata fields ({len(metadata)}) in {json_path}, skipping.")
                    continue
        except Exception as e:
            # print(f"[WARNING] Error reading {json_path}: {e}")
            continue
        
        # check modality
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
            # print(f"[WARNING] Unrecognized modality: {n}")
            continue
        
        if modality in files: # multiple scans for the same modality
            files[modality].append(p)
        else: # first scan for this modality
            files[modality] = [p]
    
    return files

# Choose slice indices based on intensity criteria
def choose_slice_indices(
    data: np.ndarray,
    label: np.ndarray,
    image_day: int,
    label_day: int,
    num_slices: int = SLICES_PER_VOLUME,
    rng: random.Random = random,
    low_ratio: float = LOW_RATIO,
    high_ratio: float = HIGH_RATIO
) -> list[int]:
    z_dim = data.shape[0]
    low = int(z_dim * low_ratio)
    high = int(z_dim * high_ratio)
    
    # slice_means = np.mean(data, axis=(0, 1))
    # middle = slice_means[low:high]
    # peak_pos = int(np.argmax(middle)) + low
    
    # peak_intensity = slice_means[peak_pos]
    # left_threshold_intensity = peak_intensity * 0.8
    # right_threshold_intensity = peak_intensity * 0.7

    # valid_indices = [i for i in range(low, peak_pos) if slice_means[i] >= left_threshold_intensity] +\
    #     [i for i in range(peak_pos, high) if slice_means[i] >= right_threshold_intensity]
    
    # Basic intensity-based filtering
    if data.shape != label.shape or volume_intensity_difference(data, label) > VOLUME_INTENSITY_THRESHOLD:
        # print("[WARNING] Volume intensity difference too high or shape mismatch.")
        return []
    
    time_diff = (label_day - image_day) // 30 # convert to months
    
    # Check for non-positive time difference
    if time_diff <= 0:
        # print("[WARNING] Non-positive time difference between image and label.")
        return []
    
    valid_indices = list(range(low, high))
    
    # Ensure enough valid slices
    if len(valid_indices) < num_slices:
        # print("[WARNING] Not enough valid slices in the specified range.")
        return []
    
    indices = []
    
    # Filter slices based on intensity difference with label
    for idx in valid_indices:
        image_slice = data[idx]
        label_slice = label[idx]
        
        # check intensity difference
        if slice_intensity_difference(image_slice, label_slice) < SLICE_INTENSITY_THRESHOLD:
            indices.append(idx)
    
    # Ensure enough valid slices after filtering
    if len(indices) < num_slices:
        # print("[WARNING] Not enough valid slices after intensity difference filtering.")
        return []
    
    indices = sorted(rng.sample(indices, num_slices)) # randomly sample desired number of slices
    
    return indices

def volume_intensity_difference(src_volume: np.ndarray, tgt_volume: np.ndarray) -> float:
    difference = np.abs(src_volume - tgt_volume)
    difference_mean = np.mean(difference)
    
    return difference_mean

def slice_intensity_difference(src_slice: np.ndarray, tgt_slice: np.ndarray) -> float:
    difference = np.abs(src_slice - tgt_slice)
    difference_mean = np.mean(difference)
    
    return difference_mean

def make_instruction(modality: str, months: int, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    instruction = template.format(MODALITY=modality, MONTHS=months)
    
    return instruction

_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIR: Path | None = None
_WORKER_SEED: int = 0

def _init_worker(uids: list[str], output_dir: str, seed: int) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIR, _WORKER_SEED
    
    _WORKER_UIDS = uids
    _WORKER_OUT_DIR = Path(output_dir)
    _WORKER_SEED = seed

def _generate_sample(idx: int) -> None:
    if _WORKER_OUT_DIR is None:
        raise RuntimeError("Worker not initialized")
    
    rng = random.Random(_WORKER_SEED + idx)
    case_id = _WORKER_UIDS[idx]
    case_files = get_oasis3_case(case_id, OASIS3_DIR)
    
    for modality in list(case_files.keys()):
        file_paths = case_files[modality]
        
        # Sort files by timepoint (day number)
        sorted_files = sorted(file_paths, key=extract_days)
        unique_files = []
        seen_days = set()
        
        # Filter to keep only one scan per unique day
        for p in sorted_files:
            d = extract_days(p)
            
            if d == 0:
                continue
            
            if d in seen_days:
                continue
            
            seen_days.add(d)
            unique_files.append(p)
        
        # Need at least 2 timepoints to create a pair
        if len(unique_files) < 2:
            # print(f"[WARNING] Not enough unique timepoints for {case_id}/{modality}: {len(unique_files)}")
            continue
        
        # Limit maximum attempts to avoid infinite loops
        max_attempts = len(unique_files)
        attempts = 0
        
        # Create unique pairs (earlier timepoint -> later timepoint)
        while len(unique_files) >= 2 and attempts < max_attempts:
            attempts += 1
            
            try:
                i, j = sorted(rng.sample(range(len(unique_files)), 2)) # indices of two different timepoints
                input_file, label_file = unique_files[i], unique_files[j]
                
                # get days
                image_day = extract_days(input_file)
                label_day = extract_days(label_file)
                
                # Load NIfTI files
                image_volume, img_header_text = load_nifti_image(input_file)
                label_volume, label_header_text = load_nifti_image(label_file)
                image_volume = image_volume.astype(np.float32)
                label_volume = label_volume.astype(np.float32)
                
                # choose slice indices
                slices = choose_slice_indices(image_volume, label_volume, image_day, label_day, num_slices=SLICES_PER_VOLUME, rng=rng, low_ratio=LOW_RATIO, high_ratio=HIGH_RATIO)
                
                if not slices:
                    continue
                
                # get mean and std for normalization
                image_mean, image_std = get_mean_and_std(image_volume)
                label_mean, label_std = get_mean_and_std(label_volume)
                
                # # Normalize image and label volumes
                # image_volume, _, _ = percentile_normalization(
                #     img_volume=image_volume,
                #     modality=modality,
                #     use_mask=True,
                # )
                # label_volume, _, _ = percentile_normalization(
                #     img_volume=label_volume,
                #     modality=modality,
                #     use_mask=True,
                # )
                
                # # Make it to uint8 for saving
                # image_volume = np.round(np.clip(image_volume, 0, 1) * 255.0).astype(np.uint8)
                # label_volume = np.round(np.clip(label_volume, 0, 1) * 255.0).astype(np.uint8)
                
                # # Resize and pad
                # image_volume = resize_and_pad(image_volume, fill_value=0).astype(np.uint8)
                # label_volume = resize_and_pad(label_volume, target_size=512, fill_value=0).astype(np.uint8)
                
                # Load metadata associated with the source image
                json_path = Path(str(input_file).replace(".nii.gz", ".json"))
                json_text = json_path.read_text()
                
                # Process each selected slice
                for slice_idx in slices:
                    image_slice = image_volume[slice_idx]
                    label_slice = label_volume[slice_idx]
                    
                    # normalize slice
                    image_slice = (image_slice - image_mean) / (image_std + 1e-8)
                    label_slice = (label_slice - label_mean) / (label_std + 1e-8)
                    
                    time_diff = (label_day - image_day) // 30 # convert to months
                    
                    instruction = make_instruction(modality, months=time_diff, rng=rng)
                    
                    data = {
                        'image': image_slice,
                        'label': label_slice,
                        'instruction': instruction,
                        'text': np.array(json_text, dtype=object),
                        "image_header": np.array(img_header_text, dtype=object)
                    }
                    
                    out_path = _WORKER_OUT_DIR / f"oasis3_{case_id}_{modality}_d{image_day:04d}_to_d{label_day:04d}_s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                    savemat(out_path, data)
                
                # Remove used files to avoid reusing them
                del unique_files[j]
                del unique_files[i]
            
            except Exception as e:
                # print(f"[ERROR] {case_id}/{modality} {input_file.name}->{label_file.name}: {e}")
                continue

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    uids = find_all_uids(Path(OASIS3_DIR) / "image")
    
    if not uids:
        raise RuntimeError(f"No OASIS3 cases found under {OASIS3_DIR}")
    
    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(len(uids))), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
    
    count = len(list(out_dir.glob("*.mat")))
    print(f"Saved {count} samples to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
