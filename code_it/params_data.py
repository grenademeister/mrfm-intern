import os

default_root: str = "/fast_storage/intern/data/instruction_tuning"

DATA_ROOT: str | None = os.environ.get("DATA_ROOT")
DATA_ROOTS: str | None = os.environ.get("DATA_ROOTS")
TRAIN_ITER: int = int(os.environ.get("TRAIN_ITER", 1))  # noqa: PLW1508


def _has_split_dirs(root: str) -> bool:
    return any(os.path.isdir(os.path.join(root, split)) for split in ("train", "val", "test"))


def _expand_root(root: str) -> list[str]:
    if _has_split_dirs(root):
        return [root]
    if not os.path.isdir(root):
        return []
    subdirs = [os.path.join(root, name) for name in os.listdir(root)]
    subdirs = [path for path in subdirs if os.path.isdir(path) and _has_split_dirs(path)]
    return sorted(subdirs)


def _get_dataset_roots() -> list[str]:
    if DATA_ROOTS:
        roots = [part.strip() for part in DATA_ROOTS.split(",") if part.strip()]
        return roots
    if DATA_ROOT:
        expanded = _expand_root(DATA_ROOT)
        return expanded if expanded else [DATA_ROOT]
    expanded = _expand_root(default_root)
    return expanded if expanded else [default_root]


DATASET_ROOTS: list[str] = _get_dataset_roots()

TRAIN_DATASET: list[str] = [root + "/train" for root in DATASET_ROOTS]
TRAIN_DATASET = TRAIN_DATASET * TRAIN_ITER
VALID_DATASET: list[str] = [root + "/val" for root in DATASET_ROOTS]
TEST_DATASET: list[str] = [root + "/test" for root in DATASET_ROOTS]
