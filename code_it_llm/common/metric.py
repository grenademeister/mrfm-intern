import torch
from torch.nn import functional

IMG_DIM: int = 4
EPS = 1e-6


class SSIMcal(torch.nn.Module):
    def __init__(
        self,
        win_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        np = win_size**2
        self.cov_norm = np / (np - 1)

    def forward(
        self,
        img: torch.Tensor,
        ref: torch.Tensor,
        data_range: torch.Tensor,
    ) -> torch.Tensor:
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = functional.conv2d(img, self.w.to(img.device))
        uy = functional.conv2d(ref, self.w.to(img.device))
        uxx = functional.conv2d(img * img, self.w.to(img.device))
        uyy = functional.conv2d(ref * ref, self.w.to(img.device))
        uxy = functional.conv2d(img * ref, self.w.to(img.device))

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2

        S = (A1 * A2) / (B1 * B2)
        return torch.mean(S, dim=[2, 3], keepdim=True)


ssim_cal = SSIMcal()


def calculate_ssim(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and (mask.dim() != IMG_DIM):
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    if mask is None:
        img_mask = img
        ref_mask = ref
    else:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)
        img_mask = img * mask
        ref_mask = ref * mask

    ones = torch.ones(ref.shape[0], device=ref.device)
    ssim = ssim_cal.forward(img_mask, ref_mask, ones)
    return ssim


def calculate_psnr(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and mask.dim() != IMG_DIM:
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    if mask is not None:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)

        img_mask = img * mask
        ref_mask = ref * mask

        mse = torch.sum((img_mask - ref_mask) ** 2, dim=(1, 2, 3)) / torch.sum(mask, dim=(1, 2, 3))
    else:
        mse = torch.mean(functional.mse_loss(img, ref, reduction="none"), dim=(1, 2, 3), keepdim=True)

    img_max = torch.amax(ref, dim=(1, 2, 3), keepdim=True) + EPS
    psnr = 10 * torch.log10(img_max**2 / (mse + EPS))
    return psnr
