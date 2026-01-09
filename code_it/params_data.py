import os

default_root: str = "/fast_storage/intern/data/instruction_tuning/brats_segmentation_mat"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
TRAIN_ITER: int = int(os.environ.get("TRAIN_ITER", 1))  # noqa: PLW1508

TRAIN_DATASET: list[str] = [
    DATA_ROOT + "/train",
]
TRAIN_DATASET = TRAIN_DATASET * TRAIN_ITER
VALID_DATASET: list[str] = [
    DATA_ROOT + "/val",
]
TEST_DATASET: list[str] = [
    DATA_ROOT + "/test",
]
