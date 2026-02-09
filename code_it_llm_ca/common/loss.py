import torch
from torch.nn import functional


def grad_loss(x, y, device="cuda"):
    mean = 0
    cx = [[[[1, -1]]]]
    cy = [[[[1], [-1]]]]
    cx = torch.FloatTensor(cx).to(device=device, dtype=torch.float32)
    cy = torch.FloatTensor(cy).to(device=device, dtype=torch.float32)
    for i in range(x.shape[1]):
        x1 = x[:, i : i + 1, :, :]
        y1 = y[:, i : i + 1, :, :]
        xx = functional.conv2d(x1, cx, padding=1)
        xy = functional.conv2d(x1, cy, padding=1)
        yx = functional.conv2d(y1, cx, padding=1)
        yy = functional.conv2d(y1, cy, padding=1)
        mean += 0.5 * (torch.mean(torch.abs(xx - yx)) + torch.mean(torch.abs(xy - yy)))
    return mean
