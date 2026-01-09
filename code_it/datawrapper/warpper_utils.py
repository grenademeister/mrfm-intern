import torch
from torch import Tensor


def pad_square(
    img: Tensor,
    target_size: int | None = None,
) -> Tensor:
    img_dims = img.dim()
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D tensor.")

    if img_dims == 3:
        img = img.unsqueeze(0)

    _, _, h, w = img.shape

    size = max(h, w)
    if target_size is not None:
        size = max(size, target_size)

    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padding = (pad_w, size - w - pad_w, pad_h, size - h - pad_h)
    img = torch.nn.functional.pad(img, padding, mode="constant", value=0)

    if img_dims == 3:
        img = img.squeeze(0)

    return img


def interpolate_to_target_width(
    img: Tensor,
    target_size: int,
) -> Tensor:
    img_dims = img.dim()
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D tensor.")

    if img_dims == 3:
        img = img.unsqueeze(0)

    _, _, h, w = img.shape
    if h > w:
        img = torch.nn.functional.interpolate(
            img,
            size=(target_size, int(w * target_size / h)),
            mode="bilinear",
            align_corners=False,
        )
    else:
        img = torch.nn.functional.interpolate(
            img,
            size=(int(h * target_size / w), target_size),
            mode="bilinear",
            align_corners=False,
        )

    if img_dims == 3:
        img = img.squeeze(0)

    return img


def pad_for_min_size(
    img: Tensor,
    min_size: int,
) -> Tensor:
    img_dims = img.dim()
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D tensor.")

    if img_dims == 3:
        img = img.unsqueeze(0)

    _, _, h, w = img.shape
    if h >= min_size and w >= min_size:
        if img_dims == 3:
            img = img.squeeze(0)
        return img

    if h < min_size:
        img = torch.nn.functional.interpolate(
            img,
            size=(min_size, int(w * min_size / h)),
            mode="bilinear",
            align_corners=False,
        )
        _, _, h, w = img.shape

    if w < min_size:
        img = torch.nn.functional.interpolate(
            img,
            size=(int(h * min_size / w), min_size),
            mode="bilinear",
            align_corners=False,
        )

    if img_dims == 3:
        img = img.squeeze(0)

    return img


def resize_512(
    img: Tensor,
) -> Tensor:
    img_dims = img.dim()
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D tensor.")

    if img_dims == 3:
        img = img.unsqueeze(0)

    _, _, h, w = img.shape
    if h == 512 and w == 512:
        if img_dims == 3:
            img = img.squeeze(0)
        return img

    if h > 512 or w > 512:
        img = interpolate_to_target_width(img, target_size=512)

    img = pad_square(img, target_size=512)

    if img_dims == 3:
        img = img.squeeze(0)

    return img
