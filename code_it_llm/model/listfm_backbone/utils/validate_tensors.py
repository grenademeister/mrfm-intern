from torch import Tensor


def validate_tensors(tensors: list[Tensor]) -> None:
    for i, t in enumerate(tensors):
        if not isinstance(t, Tensor):
            raise TypeError(f"Tensor at index {i} is not a torch.Tensor, got {type(t)} instead.")


def validate_tensor_dimensions(tensors: list[Tensor], expected_dim: int) -> None:
    for i, t in enumerate(tensors):
        if t.dim() != expected_dim:
            raise ValueError(f"Tensor at index {i} has {t.dim()} dimensions, expected {expected_dim} dimensions.")


def validate_tensor_channels(tensor: Tensor, expected_channels: int) -> None:
    if tensor.shape[1] != expected_channels:
        raise ValueError(f"Expected tensor with {expected_channels} channels, but got {tensor.shape[1]} channels.")
