import torch
from torch import Tensor


def ifft2c(img_k: Tensor) -> Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def fft2c(img: Tensor) -> Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


def apply_fixed_mask(
    img: Tensor,
    acs_num: int,
    parallel_factor: int,
) -> tuple[
    Tensor,
    Tensor,
    None,
]:
    if img.dim() != 3:
        raise ValueError("Input image must be a 3D tensor.")

    acs_half = acs_num // 2
    img_k = fft2c(img)
    C, H, W = img.shape

    mask = torch.zeros([C, H, W], dtype=torch.complex64)
    cen = mask.shape[2] // 2
    mask[:, :, cen - acs_half : cen + acs_half] = 1
    mask[:, :, ::parallel_factor] = 1
    mask = mask.to(img.device)

    output = ifft2c(img_k * mask)

    mask = mask.type(torch.float32)

    return (
        output,
        mask.type(torch.float32),
        None,
    )
