import os

# # Limit threading in numerical libraries to prevent oversubscription
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import time
import re
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import savemat

from slice_cache import DEFAULT_CACHE_MODALITY, choose_slice_index, choose_slice_index_cached, load_slice_cache

# Configuration (edit these values as needed)
DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats"
# OUTPUT_DIR = "./brats_denoise_mat"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/brats_denoise_mat"
NUM_SAMPLES = 8000
SEED = 42
WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Sampling behavior
PREFER_TUMOR_SLICE_PROB = 0.5
VALID_MODALITIES = ["flair", "t1", "t1ce", "t2"]
CACHE_MODALITY = DEFAULT_CACHE_MODALITY

# Noise parameters
NOISE_LEVEL_RANGE = (0.02, 0.15)  # Fraction of k-space magnitude
SNR_RANGE = (5.0, 25.0)  # Target SNR in dB

INSTRUCTION_TEMPLATES = [
    "Denoise this {modality} brain MRI slice.",
    "Remove noise from this {modality} MRI image.",
    "Clean up this noisy {modality} brain MRI slice.",
    "Restore this degraded {modality} MRI scan slice.",
    "Enhance the quality of this {modality} brain MRI image.",
    "Reduce artifacts in this {modality} MRI slice.",
    "Improve the signal quality of this {modality} brain MRI slice.",
    "Suppress noise in this {modality} MRI scan.",
    "Reconstruct a clean {modality} image from this noisy input.",
    "Filter out noise from this {modality} brain MRI slice.",
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


def add_kspace_noise(image: np.ndarray, target_snr: float, rng: random.Random = random) -> np.ndarray:
    """Add noise in k-space to achieve target SNR and return noisy image."""
    # Calculate signal level from image
    mask = image > 0.05 * np.max(image)
    signal_level = image[mask].mean()

    # Calculate noise standard deviation from target SNR (in image space)
    noise_std_image = signal_level / target_snr

    # Scale for k-space: need to account for IFFT normalization
    # numpy's ifft2 divides by N, so k-space noise needs to be scaled up
    n_pixels = image.shape[0] * image.shape[1]
    noise_std_kspace = noise_std_image * np.sqrt(n_pixels)

    # Convert to k-space
    kspace = np.fft.fft2(image)

    # Create numpy RNG from random.Random seed state
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    # Add complex Gaussian noise in k-space
    noise_real = np_rng.normal(0, noise_std_kspace, kspace.shape)
    noise_imag = np_rng.normal(0, noise_std_kspace, kspace.shape)
    noisy_kspace = kspace + noise_real + 1j * noise_imag

    # Convert back to image space
    noisy_image = np.fft.ifft2(noisy_kspace)

    # Return magnitude (MRI images are magnitude images)
    return np.abs(noisy_image).astype(np.float32)


def calculate_snr(clean_image: np.ndarray, noisy_image: np.ndarray) -> float:
    """Calculate SNR between clean and noisy images."""
    mask = clean_image > 0.05 * np.max(clean_image)
    signal = clean_image[mask].mean()
    noise = noisy_image[mask] - clean_image[mask]
    noise_std = np.std(noise)
    if noise_std == 0:
        return float("inf")
    return signal / noise_std


def make_instruction(modality: str, rng: random.Random = random) -> str:
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return template.format(modality=modality.upper())


_WORKER_UIDS: list[str] = []
_WORKER_OUT_DIR: Path | None = None
_WORKER_SEED = 0
_WORKER_CACHE: dict[str, tuple[np.ndarray, np.ndarray, int]] | None = None


def _init_worker(uids: list[str], out_dir: str, seed: int) -> None:
    global _WORKER_UIDS, _WORKER_OUT_DIR, _WORKER_SEED, _WORKER_CACHE
    _WORKER_UIDS = uids
    _WORKER_OUT_DIR = Path(out_dir)
    _WORKER_SEED = seed
    _WORKER_CACHE = load_slice_cache(CACHE_MODALITY)


def _generate_sample(idx: int) -> None:
    if _WORKER_OUT_DIR is None:
        raise RuntimeError("Worker not initialized")
    rng = random.Random(_WORKER_SEED + idx)
    case_id = rng.choice(_WORKER_UIDS)
    case_files = get_brats_case(case_id, BRATS_DIR)

    modality = rng.choice(VALID_MODALITIES)
    img_volume, header_text = load_nifti_data_and_header(case_files[modality])
    slice_idx = choose_slice_index_cached(
        case_id,
        img_volume.shape[2],
        PREFER_TUMOR_SLICE_PROB,
        rng,
        _WORKER_CACHE,
        lambda: choose_slice_index(
            img_volume,
            load_nifti_image(case_files["seg"]),
            PREFER_TUMOR_SLICE_PROB,
            rng,
        ),
    )
    clean_slice = img_volume[:, :, slice_idx].astype(np.float32)

    # Generate noisy version via k-space with target SNR
    target_snr = rng.uniform(*SNR_RANGE)
    noisy_slice = add_kspace_noise(clean_slice, target_snr, rng=rng)

    instruction = make_instruction(modality, rng=rng)

    data = {
        "image": noisy_slice,
        "label": clean_slice,
        "instruction": np.array(instruction, dtype=object),
        "text": np.array(header_text, dtype=object),
    }

    out_path = _WORKER_OUT_DIR / f"brats_{case_id}_{modality}_denoise_slice_{slice_idx:03d}_{idx:06d}.mat"
    savemat(out_path, data)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = find_all_uids(Path(BRATS_DIR) / "seg")
    if not uids:
        raise RuntimeError(f"No BRATS cases found under {BRATS_DIR}")

    start_time = time.time()
    log_every = max(1, NUM_SAMPLES // 100)

    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(uids, str(out_dir), SEED)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_generate_sample, range(NUM_SAMPLES)), start=1):
            if done % log_every == 0 or done == NUM_SAMPLES:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = NUM_SAMPLES - done
                eta_min = (remaining / rate) / 60 if rate > 0 else float("inf")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{timestamp}] {done}/{NUM_SAMPLES} samples created | " f"{rate:.2f}/s | ETA {eta_min:.1f} min",
                    flush=True,
                )

    print(f"\nSaved {NUM_SAMPLES} samples to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
