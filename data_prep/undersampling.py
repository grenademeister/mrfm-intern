import numpy as np
from typing import Tuple, Optional


def fft2c(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Centered FFT on last two dims for numpy arrays.

    Args:
        x: ndarray with shape (..., H, W) or (C, H, W)
        norm: normalization for fft (default: "ortho")

    Returns:
        k-space ndarray with same shape and dtype (complex) as FFT result.
    """
    if x.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (H, W) on the last axes.")

    x = np.fft.ifftshift(x, axes=(-2, -1))
    k = np.fft.fftn(x, axes=(-2, -1), norm=norm)
    k = np.fft.fftshift(k, axes=(-2, -1))

    return k


def ifft2c(k: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Centered IFFT on last two dims for numpy arrays."""
    if k.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (H, W) on the last axes.")

    k = np.fft.ifftshift(k, axes=(-2, -1))
    x = np.fft.ifftn(k, axes=(-2, -1), norm=norm)
    x = np.fft.fftshift(x, axes=(-2, -1))

    return x


def apply_fixed_mask(
    img: np.ndarray,  # (C, H, W)
    acs_num: int,
    parallel_factor: int,
    norm: str = "ortho",
) -> Tuple[np.ndarray, np.ndarray, None]:
    """Apply a fixed ACS + uniform undersampling mask and return image, mask, None.

    Args:
        img: numpy array with shape (C, H, W)
        acs_num: number of central k-space columns to keep (ACS)
        parallel_factor: acceleration factor (keep every R-th column)
        norm: normalization for fft/ifft

    Returns:
        output: reconstructed image after applying mask (numpy ndarray)
        mask: mask applied (C, H, W) with dtype float32
    """
    if img.ndim != 3:
        raise ValueError("Expected img with shape (C, H, W).")
    if parallel_factor <= 0:
        raise ValueError("parallel_factor must be >= 1.")
    if acs_num < 0:
        raise ValueError("acs_num must be >= 0.")

    C, H, W = img.shape

    # k-space
    img_k = fft2c(img, norm=norm)

    # build a shared (H, W) mask, then broadcast to coils
    mask2d = np.zeros((H, W), dtype=np.float32)

    # exact ACS width (works for odd/even)
    acs_num = min(acs_num, W)
    start = (W - acs_num) // 2
    end = start + acs_num
    mask2d[:, start:end] = 1.0

    # uniform undersampling (keep every R-th column)
    if parallel_factor > 1:
        mask2d[:, ::parallel_factor] = 1.0
    else:
        mask2d[:] = 1.0

    mask = np.broadcast_to(mask2d[None, ...], (C, H, W)).astype(np.float32)

    # apply mask in k-space (ensure dtype compatibility)
    output = ifft2c(img_k * mask.astype(img_k.dtype), norm=norm)

    return output, mask
