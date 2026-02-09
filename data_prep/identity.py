import os
import random
import re
from pathlib import Path
import json

import scipy.io as sio
import nibabel as nib
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim


DATA_DIR = "/fast_storage/intern/data/data_curation"
OASIS3_DIR = f"{DATA_DIR}/oasis3"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/oasis3_identity_mat"
SEED = 42
WORKERS = max(1, (os.cpu_count() or 1) - 15)

INSTRUCTION_TEMPLATES = [
    "Generate the exact same {MODALITY} brain MRI slice as the input.",
    "Reconstruct this {MODALITY} MRI image without any changes.",
    "Output the identity mapping of this {MODALITY} brain MRI scan.",
    "Maintain the original structures of this {MODALITY} slice perfectly.",
    "Produce an identical {MODALITY} MRI slice from the given input.",
    "Keep this {MODALITY} brain MRI slice unchanged.",
    "Synthesize the same {MODALITY} image as provided, ensuring zero modification.",
    "Return the input {MODALITY} brain MRI slice as it is."
]

SLICES_PER_VOLUME = 1
LOW_RATIO = 0.4
HIGH_RATIO = 0.7
SSIM_THRESHOLD = 0.6

def extract_days(path: str | Path) -> int:
    name = path.name if isinstance(path, Path) else path
    match = re.search(r"sess?-d(\d+)", name)  # matches d0, d100, etc.
    return int(match.group(1)) if match else 0


def load_nifti_image(file_path: Path) -> tuple[np.ndarray, str]:
    img = nib.load(str(file_path))
    img = nib.as_closest_canonical(img)  # reorient to RAS
    data = np.asanyarray(img.dataobj).astype(np.float32)

    assert data.ndim == 4 or data.ndim == 3, f"Expected 4D or 3D data, but shape of data is {data.shape}."

    # multi-coil case: (H, W, C, T)
    if data.ndim == 4:
        data = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1))  # magnitude-only RSS

    data = data.transpose(2, 0, 1)  # (C, H, W)
    data = np.ascontiguousarray(np.rot90(data, k=1, axes=(1, 2)))  # correct rotation

    return data, img.header.__str__()


def find_all_uids(root: str | Path) -> list[str]:
    root = Path(root)
    pat = re.compile(r"oasis3_sub-OAS(\d+)_")  # matches oasis3_sub-OAS<digits>_
    uids: set[str] = set()

    for p in root.glob("*.nii.gz"):
        m = pat.search(p.name)
        if m:
            uids.add(m.group(1))

    return sorted(uids, key=int)


def get_oasis3_case(case_id: str, root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    modality_root = root / "image"

    files: dict[str, list[Path]] = {}
    pattern = f"*OAS{case_id}_*.nii.gz"

    for p in modality_root.glob(pattern):
        n = p.name.upper()

        # Skip multi-echo sequences (keep only primary echoes)
        if "ECHO-" in n:
            continue

        # Check for corresponding JSON metadata file
        json_path = p.parent / (p.name.replace(".nii.gz", ".json"))
        if not json_path.exists():
            continue

        # Validate metadata quality
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                if len(metadata) < 15:
                    continue
        except Exception:
            continue

        # check modality
        if "T1" in n and ("POST" in n or "CONTRAST" in n):
            modality = "t1ce"
        elif "FLAIR" in n:
            modality = "flair"
        elif "T2" in n and ("POST" in n or "CONTRAST" in n):  # skip T2 POST if exists
            continue
        elif "T2" in n:
            modality = "t2"
        elif "T1" in n:
            modality = "t1"
        else:
            continue

        files.setdefault(modality, []).append(p)

    return files


def choose_slice_indices_simple(
    z_dim: int,
    num_slices: int = SLICES_PER_VOLUME,
    rng: random.Random = random,
    low_ratio: float = LOW_RATIO,
    high_ratio: float = HIGH_RATIO,
) -> list[int]:
    low = int(z_dim * low_ratio)
    high = int(z_dim * high_ratio)
    if high <= low or (high - low) < num_slices:
        return []
    
    candidates = list(range(low, high))
    return sorted(rng.sample(candidates, num_slices))


_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIR: Path | None = None
_WORKER_SEED: int = 0


def _init_worker(uids: list[str], output_dir: str, seed: int) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIR, _WORKER_SEED
    _WORKER_UIDS = uids
    _WORKER_OUT_DIR = Path(output_dir)
    _WORKER_SEED = seed


def make_instruction(modality: str, months: int, rng=random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return template.format(MODALITY=modality, MONTHS=months)


def choose_slice_indices_fast(
    data: np.ndarray,
    label: np.ndarray,
    image_day: int,
    label_day: int,
    num_slices: int = SLICES_PER_VOLUME,
    rng: random.Random = random,
    low_ratio: float = LOW_RATIO,
    high_ratio: float = HIGH_RATIO,
) -> list[int]:
    if data.shape != label.shape:
        return []

    months = (label_day - image_day) // 30
    if months <= 0:
        return []

    z_dim = data.shape[0]
    low = int(z_dim * low_ratio)
    high = int(z_dim * high_ratio)

    if high <= low:
        return []

    candidates = list(range(low, high))
    if len(candidates) < num_slices:
        return []

    rng.shuffle(candidates)

    picked: list[int] = []

    for idx in candidates:
        dr = float(label[idx].max() - label[idx].min())
        if dr <= 1e-6:
            continue
        if ssim(data[idx], label[idx], data_range=dr) > SSIM_THRESHOLD:
            picked.append(idx)
            if len(picked) >= num_slices:
                break

    if len(picked) < num_slices:
        return []

    return sorted(picked)


def _generate_sample(idx: int) -> int:
    if _WORKER_OUT_DIR is None:
        raise RuntimeError("Worker not initialized")

    rng = random.Random(_WORKER_SEED + idx)
    case_id = _WORKER_UIDS[idx]
    case_files = get_oasis3_case(case_id, OASIS3_DIR)

    saved = 0

    for modality, file_paths in case_files.items():
        for input_file in file_paths:
            try:
                image_day = extract_days(input_file)
                image_volume, img_header_text = load_nifti_image(input_file)

                slices = choose_slice_indices_simple(
                    image_volume.shape[0],
                    num_slices=SLICES_PER_VOLUME,
                    rng=rng
                )
                
                if not slices:
                    continue

                json_path = Path(str(input_file).replace(".nii.gz", ".json"))
                json_text = json_path.read_text(encoding="utf-8") if json_path.exists() else "{}"
                
                instruction = rng.choice(INSTRUCTION_TEMPLATES).format(MODALITY=modality)

                for slice_idx in slices:
                    image_slice = image_volume[slice_idx].astype(np.float32, copy=False)
                    label_slice = image_slice 

                    out_path = _WORKER_OUT_DIR / (
                        f"oasis3_{case_id}_{modality}_d{image_day:04d}_identity_"
                        f"s{slice_idx + 1:03d}_n{idx + 1:05d}.mat"
                    )
                    
                    if out_path.exists():
                        continue

                    data = {
                        "image": image_slice,
                        "label": label_slice,
                        "instruction": np.array(instruction, dtype=object),
                        "text": np.array(json_text, dtype=object),
                        "image_header": np.array(img_header_text, dtype=object),
                    }
                    sio.savemat(out_path, data)
                    saved += 1

            except Exception:
                continue

    return saved


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(OASIS3_DIR) / "image")
    if not uids:
        raise RuntimeError(f"No OASIS3 cases found under {OASIS3_DIR}/image")

    total = len(uids)
    print(f"Total subjects: {total}, workers: {WORKERS}")

    chunksize = 4 if total < 200 else 8

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(total), chunksize=chunksize), start=1):
            print(f"{done}/{len(uids)} subjects done", end="\r")
    
    count = len(list(out_dir.glob("*.mat")))
    print(f"Saved {count} samples to {out_dir.resolve()}")

if __name__ == "__main__":
    main()