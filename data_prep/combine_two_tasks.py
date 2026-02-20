import os
import random
from multiprocessing import Pool
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat, savemat


# Configuration (edit these values as needed)
SRC1_DIR = "/fast_storage/intern/data/instruction_tuning/multi_task/denoising"
SRC2_DIR = "/fast_storage/intern/data/instruction_tuning/multi_task/crossmodal"
OUTPUT_DIR = "/fast_storage/intern/data/instruction_tuning/multi_task/denoising_and_crossmodal"
SEED = 42
STRICT = False
WORKERS = max(1, (os.cpu_count() or 1) - 25)
# Optional per-split cap (set to None for no limit)
SPLIT_LIMITS = {
    "train": 5000,
    "val": 300,
    "test": 300,
}

DEFAULT_CONNECTORS = [
    "and",
    "then",
    "after that",
    "and then",
    "afterwards",
    "and afterward",
    "followed by",
    "then after that",
]

# FASTMRI_NAME_RE = re.compile(r"fastmri_(\d+)_.*_s(\d+)_n\d+\.mat$", re.IGNORECASE)
BRATS_CROSS_RE = re.compile(
    r"brats2023_(?P<tumor>gli|men)_(?P<pid>\d+)_(?P<src>t1|t2)_to_(?P<tgt>t1|t2)_s(?P<slice>\d+)_n\d+\.mat$",
    re.IGNORECASE,
)
BRATS_ACCEL_RE = re.compile(
    r"brats2023_(?P<tumor>gli|men)_(?P<pid>\d+)_(?P<mod>t1|t2)_r\d+_s(?P<slice>\d+)_n\d+\.mat$",
    re.IGNORECASE,
)
BRATS_DENOISE_RE = re.compile(
    r"brats2023_(?P<tumor>gli|men)_(?P<pid>\d+)_(?P<mod>t1|t2)_sigma[0-9.]+_s(?P<slice>\d+)_n\d+\.mat$",
    re.IGNORECASE,
)
BRATS_SEG_RE = re.compile(
    r"brats2023_(?P<tumor>gli|men)_(?P<pid>\d+)_(?P<mod>t1|t2)_s(?P<slice>\d+)_n\d+\.mat$",
    re.IGNORECASE,
)


def _extract_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            try:
                return str(value.item())
            except Exception:
                return str(value)
        # handle char arrays or object arrays
        flat = value.ravel()
        try:
            return "".join(str(x) for x in flat)
        except Exception:
            return str(value)
    return str(value)


def _get_key(data: dict, keys: list[str]):
    for k in keys:
        if k in data:
            return data[k]
    return None


def _combine_instruction(inst1: str, inst2: str, connector: str) -> str:
    inst1 = inst1.strip()
    inst2 = inst2.strip()
    if not inst1 and not inst2:
        return ""
    if inst1 and not inst2:
        return inst1
    if inst2 and not inst1:
        return inst2
    if inst1.endswith("."):
        inst1 = inst1[:-1].rstrip()
    if inst2:
        inst2 = inst2[0].lower() + inst2[1:]
    return f"{inst1} {connector} {inst2}"

def _parse_fastmri_key(path: Path) -> tuple[str, str, str, int] | None:
    name = path.name
    m = BRATS_CROSS_RE.match(name)
    if m:
        return (
            m.group("tumor").lower(),
            m.group("pid"),
            m.group("src").lower(),  # use src modality for matching
            int(m.group("slice")),
        )
    m = BRATS_ACCEL_RE.match(name)
    if m:
        return (
            m.group("tumor").lower(),
            m.group("pid"),
            m.group("mod").lower(),
            int(m.group("slice")),
        )
    m = BRATS_DENOISE_RE.match(name)
    if m:
        return (
            m.group("tumor").lower(),
            m.group("pid"),
            m.group("mod").lower(),
            int(m.group("slice")),
        )
    m = BRATS_SEG_RE.match(name)
    if m:
        return (
            m.group("tumor").lower(),
            m.group("pid"),
            m.group("mod").lower(),
            int(m.group("slice")),
        )
    return None

_WORKER_FILES1: list[Path] = []
_WORKER_SRC2_BY_KEY: dict[tuple[str, str, str, int], list[Path]] = {}
_WORKER_OUT_SPLIT: Path | None = None
_WORKER_SEED = 0
_WORKER_STRICT = False
_WORKER_SPLIT_NAME = ""


def _init_worker(
    files1: list[Path],
    src2_by_key: dict[tuple[str, str, int], list[Path]],
    out_split: Path,
    seed: int,
    strict: bool,
    split_name: str,
) -> None:
    global _WORKER_FILES1, _WORKER_SRC2_BY_KEY, _WORKER_OUT_SPLIT
    global _WORKER_SEED, _WORKER_STRICT, _WORKER_SPLIT_NAME
    _WORKER_FILES1 = files1
    _WORKER_SRC2_BY_KEY = src2_by_key
    _WORKER_OUT_SPLIT = out_split
    _WORKER_SEED = seed
    _WORKER_STRICT = strict
    _WORKER_SPLIT_NAME = split_name


def _process_index(idx: int) -> tuple[int, int]:
    if _WORKER_OUT_SPLIT is None:
        raise RuntimeError("Worker not initialized")

    p1 = _WORKER_FILES1[idx]
    key = _parse_fastmri_key(p1)
    if key is None:
        raise ValueError(f"[{_WORKER_SPLIT_NAME}] Unrecognized filename: {p1.name}")

    candidates = _WORKER_SRC2_BY_KEY.get(key)
    if not candidates:
        if _WORKER_STRICT:
            raise FileNotFoundError(f"[{_WORKER_SPLIT_NAME}] Missing in src2 for key: {key}")
        return (0, 1)

    if len(candidates) > 1 and _WORKER_STRICT:
        raise ValueError(f"[{_WORKER_SPLIT_NAME}] Multiple src2 matches for key {key}: {candidates}")

    p2 = candidates[0]

    d1 = loadmat(p1, squeeze_me=True, struct_as_record=False)
    d2 = loadmat(p2, squeeze_me=True, struct_as_record=False)

    image = _get_key(d1, ["image", "input"])
    if image is None:
        raise KeyError(f"[{_WORKER_SPLIT_NAME}] No image/input in {p1}")

    label = _get_key(d2, ["label", "target"])
    if label is None:
        raise KeyError(f"[{_WORKER_SPLIT_NAME}] No label/target in {p2}")

    rng = random.Random(_WORKER_SEED + idx)
    inst1 = _extract_str(_get_key(d1, ["instruction"]))
    inst2 = _extract_str(_get_key(d2, ["instruction"]))
    connector = rng.choice(DEFAULT_CONNECTORS)
    instruction = _combine_instruction(inst1, inst2, connector)

    text = _get_key(d1, ["text"])
    image_header = _get_key(d1, ["image_header", "header"])

    out_data = {
        "image": image,
        "label": label,
        "instruction": np.array(instruction, dtype=object),
        "text": text if text is not None else np.array("", dtype=object),
        "image_header": image_header if image_header is not None else np.array("", dtype=object),
    }

    out_path = _WORKER_OUT_SPLIT / p1.name
    savemat(out_path, out_data)
    return (1, 0)


def combine_split(
    split_name: str,
    src1_split: Path,
    src2_split: Path,
    out_split: Path,
    strict: bool,
    workers: int,
    seed: int,
    limit: int | None = None,
) -> dict[str, int]:
    out_split.mkdir(parents=True, exist_ok=True)

    files1 = sorted(src1_split.glob("*.mat"))
    if not files1:
        return {"written": 0, "missing_in_src2": 0}
    if limit is not None:
        rng = random.Random(seed)
        if limit < len(files1):
            files1 = rng.sample(files1, limit)

    src2_files = sorted(src2_split.glob("*.mat"))
    if not src2_files:
        return {"written": 0, "missing_in_src2": len(files1)}

    src2_by_key: dict[tuple[str, str, str, int], list[Path]] = {}
    for p2 in src2_files:
        key = _parse_fastmri_key(p2)
        if key is None:
            if strict:
                raise ValueError(f"[{split_name}] Unrecognized filename in src2: {p2.name}")
            continue
        src2_by_key.setdefault(key, []).append(p2)

    # keep deterministic order for non-strict multi-match selection
    for key in src2_by_key:
        src2_by_key[key] = sorted(src2_by_key[key])

    missing_in_src2 = 0
    written = 0

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(files1, src2_by_key, out_split, seed, strict, split_name),
    ) as pool:
        total = len(files1)
        for done, (w, m) in enumerate(
            pool.imap_unordered(_process_index, range(total)), start=1
        ):
            written += w
            missing_in_src2 += m
            print(f"[{split_name}] {done}/{total} files done", end="\r")
    print()

    return {"written": written, "missing_in_src2": missing_in_src2}


def main() -> None:
    src1 = Path(SRC1_DIR)
    src2 = Path(SRC2_DIR)
    out = Path(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        src1_split = src1 / split
        src2_split = src2 / split
        out_split = out / split

        if not src1_split.exists():
            raise FileNotFoundError(f"Missing split in src1: {src1_split}")
        if not src2_split.exists():
            raise FileNotFoundError(f"Missing split in src2: {src2_split}")

        stats = combine_split(
            split_name=split,
            src1_split=src1_split,
            src2_split=src2_split,
            out_split=out_split,
            strict=STRICT,
            workers=WORKERS,
            seed=SEED,
            limit=SPLIT_LIMITS.get(split),
        )
        print(
            f"[{split}] wrote {stats['written']} files (missing in src2: {stats['missing_in_src2']})"
        )


if __name__ == "__main__":
    main()
