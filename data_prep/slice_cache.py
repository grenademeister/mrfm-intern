import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_DIR = "/fast_storage/intern/data/data_curation"
BRATS_DIR = f"{DATA_DIR}/brats"
CACHE_DIR = Path("/fast_storage/intern/data/data_curation/brats_cache")
DEFAULT_CACHE_MODALITY = "flair"
DEFAULT_THRESHOLD_RATIO = 0.5
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 10)
VALID_MODALITIES = ["flair", "t1", "t1ce", "t2"]

_WORKER_BRATS_DIR: str | None = None
_WORKER_MODALITY: str | None = None
_WORKER_THRESHOLD_RATIO: float | None = None


def find_all_uids(root: str | Path) -> list[str]:
    root = Path(root)
    uids = set()
    for p in root.glob("*.nii.gz"):
        m = re.search(r"BraTS2021_(\d+)_", p.name)
        if m:
            uids.add(m.group(1))
    return sorted(uids)


def get_brats_case(case_id: str, root: str | Path, modalities: list[str]) -> dict[str, Path]:
    root = Path(root)
    seg_root = root / "seg"
    modality_root = root / "image"

    files: dict[str, Path] = {}
    seg_pattern = f"*_{case_id}_seg.nii.gz"
    seg_matches = list(seg_root.glob(seg_pattern))
    if len(seg_matches) != 1:
        raise FileNotFoundError(f"{case_id} seg: {seg_matches}")
    files["seg"] = seg_matches[0]

    for m in modalities:
        pattern = f"*_{case_id}_{m}.nii.gz"
        matches = list(modality_root.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"{case_id} {m}: {matches}")
        files[m] = matches[0]

    return files


def cache_path(modality: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"slice_cache_{modality}.npz"


def build_slice_cache(
    brats_dir: str | Path = BRATS_DIR,
    modality: str = DEFAULT_CACHE_MODALITY,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    cache_dir: Path = CACHE_DIR,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    if modality not in VALID_MODALITIES:
        raise ValueError(f"Unknown modality {modality}. Expected one of {VALID_MODALITIES}.")

    uids = find_all_uids(Path(brats_dir) / "seg")
    if not uids:
        raise RuntimeError(f"No BRATS cases found under {brats_dir}")

    workers = max(1, workers)
    depths: list[int] = []
    valid_masks: list[np.ndarray] = []
    tumor_masks: list[np.ndarray] = []
    with Pool(processes=workers, initializer=_init_worker, initargs=(str(brats_dir), modality, threshold_ratio)) as pool:
        for done, (depth, valid_mask, tumor_mask) in enumerate(pool.imap(_compute_case_cache, uids, chunksize=1), start=1):
            depths.append(depth)
            valid_masks.append(valid_mask)
            tumor_masks.append(tumor_mask)
            print(f"{done} volumes done", flush=True)

    max_depth = max(depths)
    valid_array = np.zeros((len(uids), max_depth), dtype=np.bool_)
    tumor_array = np.zeros((len(uids), max_depth), dtype=np.bool_)

    for i, depth in enumerate(depths):
        valid_array[i, :depth] = valid_masks[i]
        tumor_array[i, :depth] = tumor_masks[i]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_path(modality, cache_dir)
    np.savez(
        out_path,
        uids=np.array(uids),
        depths=np.array(depths, dtype=np.int32),
        valid_masks=valid_array,
        tumor_masks=tumor_array,
        modality=np.array(modality),
        threshold_ratio=np.array(threshold_ratio, dtype=np.float32),
    )
    return out_path


def load_slice_cache(modality: str = DEFAULT_CACHE_MODALITY, cache_dir: Path = CACHE_DIR) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    path = cache_path(modality, cache_dir)
    if not path.exists():
        return {}

    data = np.load(path, allow_pickle=False)
    uids = data["uids"]
    depths = data["depths"]
    valid_masks = data["valid_masks"]
    tumor_masks = data["tumor_masks"]
    return {
        str(uid): (valid_masks[i, : int(depths[i])].copy(), tumor_masks[i, : int(depths[i])].copy(), int(depths[i]))
        for i, uid in enumerate(uids)
    }


def _init_worker(brats_dir: str, modality: str, threshold_ratio: float) -> None:
    global _WORKER_BRATS_DIR, _WORKER_MODALITY, _WORKER_THRESHOLD_RATIO
    _WORKER_BRATS_DIR = brats_dir
    _WORKER_MODALITY = modality
    _WORKER_THRESHOLD_RATIO = threshold_ratio


def _compute_case_cache(case_id: str) -> tuple[int, np.ndarray, np.ndarray]:
    if _WORKER_BRATS_DIR is None or _WORKER_MODALITY is None or _WORKER_THRESHOLD_RATIO is None:
        raise RuntimeError("Worker not initialized")
    case_files = get_brats_case(case_id, _WORKER_BRATS_DIR, [_WORKER_MODALITY])
    img = nib.load(str(case_files[_WORKER_MODALITY])).get_fdata(dtype=np.float32)
    slice_means = img.mean(axis=(0, 1))
    max_intensity = float(slice_means.max())
    intensity_threshold = _WORKER_THRESHOLD_RATIO * max_intensity
    valid_mask = (slice_means > intensity_threshold).astype(np.bool_)

    seg = nib.load(str(case_files["seg"])).get_fdata(dtype=np.float32)
    tumor_mask = np.any(seg != 0, axis=(0, 1)).astype(np.bool_)

    depth = int(img.shape[2])
    if tumor_mask.shape[0] != depth:
        raise RuntimeError(f"Depth mismatch for {case_id}: {depth} vs {tumor_mask.shape[0]}")

    return depth, valid_mask, tumor_mask


def choose_slice_index(
    tgt_volume: np.ndarray,
    seg_volume: np.ndarray,
    prefer_tumor_prob: float,
    rng,
) -> int:
    depth = tgt_volume.shape[2]
    slice_means = tgt_volume.mean(axis=(0, 1))
    valid_slices = np.flatnonzero(slice_means > 0.5 * float(slice_means.max()))
    if valid_slices.size == 0:
        return int(rng.randrange(depth))

    tumor_slices = np.flatnonzero(np.any(seg_volume != 0, axis=(0, 1)))
    valid_tumor_slices = np.intersect1d(valid_slices, tumor_slices)

    if tumor_slices.size > 0 and rng.random() < prefer_tumor_prob:
        return int(rng.choice(valid_tumor_slices))
    return int(rng.choice(valid_slices))


def choose_slice_index_from_cache(
    depth: int,
    valid_mask: np.ndarray | None,
    tumor_mask: np.ndarray | None,
    prefer_tumor_prob: float,
    rng,
) -> int:
    if valid_mask is None:
        return int(rng.randrange(depth))

    valid_slices = np.flatnonzero(valid_mask)
    if valid_slices.size == 0:
        return int(rng.randrange(depth))

    if tumor_mask is None:
        return int(rng.choice(valid_slices))

    valid_tumor_slices = np.flatnonzero(valid_mask & tumor_mask)
    if valid_tumor_slices.size > 0 and rng.random() < prefer_tumor_prob:
        return int(rng.choice(valid_tumor_slices))
    return int(rng.choice(valid_slices))


def choose_slice_index_cached(
    case_id: str,
    depth: int,
    prefer_tumor_prob: float,
    rng,
    cache: dict[str, tuple[np.ndarray, np.ndarray, int]] | None,
    fallback,
) -> int:
    entry = cache.get(case_id) if cache else None
    if entry and entry[2] == depth:
        valid_mask, tumor_mask, depth = entry
        return choose_slice_index_from_cache(depth, valid_mask, tumor_mask, prefer_tumor_prob, rng)
    return fallback()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BRATS slice cache.")
    parser.add_argument("--brats-dir", default=BRATS_DIR)
    parser.add_argument("--modality", default=DEFAULT_CACHE_MODALITY, choices=VALID_MODALITIES)
    parser.add_argument("--threshold-ratio", type=float, default=DEFAULT_THRESHOLD_RATIO)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = build_slice_cache(
        brats_dir=args.brats_dir,
        modality=args.modality,
        threshold_ratio=args.threshold_ratio,
        cache_dir=Path(args.cache_dir),
        workers=args.workers,
    )
    print(f"Cache saved to {out_path}")


if __name__ == "__main__":
    main()
