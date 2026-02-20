import cv2
import numpy as np
from typing import Optional
from scipy.ndimage import zoom, distance_transform_edt
from PIL import Image, ImageFilter

MODALITY_CFG = {
    "T1":    {"transform": "linear", "p": (0.5, 99.5)},
    "T1CE":  {"transform": "linear", "p": (0.5, 99.8)},
    "T2":    {"transform": "linear", "p": (1.0, 99.0)},
    "FLAIR": {"transform": "linear", "p": (0.5, 99.7)},
    "PD":    {"transform": "linear", "p": (1.0, 99.0)},
    "DWI":   {"transform": "log1p",  "p": (0.1, 99.9)},
    "ADC":   {"transform": "linear", "p": (1.0, 99.0)},
    "SWI":   {"transform": "log1p",  "p": (1.0, 99.5)},
}


def _is_binary_array(arr: np.ndarray) -> bool:
    """Heuristic to detect binary masks (values in {0,1} or dtype bool)."""
    if arr.dtype == bool:
        return True
    
    uniques = np.unique(arr)
    
    return uniques.size <= 2 and np.all(np.isin(uniques, [0, 1]))


# zero-pad to match the target size
def pad_square(
    img: np.ndarray,
    target_size: int | None = None,
) -> np.ndarray:
    img_dims = img.ndim
    
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D array.")
    
    if img_dims == 3:
        img = np.expand_dims(img, axis=0)
    
    _, _, h, w = img.shape
    
    size = max(h, w)
    
    if target_size is not None:
        size = max(size, target_size)
    
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padding = ((0, 0), (0, 0), (pad_h, size - h - pad_h), (pad_w, size - w - pad_w))
    img = np.pad(img, padding, mode="constant", constant_values=0)
    
    if img_dims == 3:
        img = np.squeeze(img, axis=0)
    
    return img


# bilinear interpolate to target size while maintaining aspect ratio
# larger dimension (height or width) is resized to target size
def interpolate_to_target_width(
    img: np.ndarray,
    target_size: int,
) -> np.ndarray:
    img_dims = img.ndim
    
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D array.")
    
    if img_dims == 3:
        img = np.expand_dims(img, axis=0)
    
    _, _, h, w = img.shape
    
    if h > w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size
    
    # Resize each channel and batch separately
    batch_size, channels = img.shape[:2]
    resized_img = np.zeros((batch_size, channels, new_h, new_w), dtype=img.dtype)
    is_binary = _is_binary_array(img)
    order = 0 if is_binary else 1  # nearest neighbor for masks, bilinear otherwise
    
    for b in range(batch_size):
        for c in range(channels):
            scale_h = new_h / h
            scale_w = new_w / w
            resized_img[b, c] = zoom(img[b, c], (scale_h, scale_w), order=order)

    if is_binary:
        resized_img = (resized_img > 0.5).astype(img.dtype)
    
    img = resized_img
    
    if img_dims == 3:
        img = np.squeeze(img, axis=0)
    
    return img


# bilinear interpolate to ensure minimum size in both dimensions
def pad_for_min_size(
    img: np.ndarray,
    min_size: int,
) -> np.ndarray:
    img_dims = img.ndim
    
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D array.")
    
    if img_dims == 3:
        img = np.expand_dims(img, axis=0)
    
    _, _, h, w = img.shape
    
    if h >= min_size and w >= min_size:
        if img_dims == 3:
            img = np.squeeze(img, axis=0)
        return img
    
    batch_size, channels = img.shape[:2]
    is_binary = _is_binary_array(img)
    order = 0 if is_binary else 1
    
    if h < min_size:
        new_h = min_size
        new_w = int(w * min_size / h)
        resized_img = np.zeros((batch_size, channels, new_h, new_w), dtype=img.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                scale_h = new_h / h
                scale_w = new_w / w
                resized_img[b, c] = zoom(img[b, c], (scale_h, scale_w), order=order)
        
        img = resized_img
        _, _, h, w = img.shape
    
    if w < min_size:
        new_h = int(h * min_size / w)
        new_w = min_size
        resized_img = np.zeros((batch_size, channels, new_h, new_w), dtype=img.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                scale_h = new_h / h
                scale_w = new_w / w
                resized_img[b, c] = zoom(img[b, c], (scale_h, scale_w), order=order)
        
        img = resized_img

    if is_binary:
        img = (img > 0.5).astype(img.dtype)
    
    if img_dims == 3:
        img = np.squeeze(img, axis=0)
    
    return img


# resize to 512x512
# if larger than 512, interpolate to 512 on larger dimension
# then zero-pad to 512x512
def resize_512(
    img: np.ndarray,
) -> np.ndarray:
    img_dims = img.ndim
    
    if img_dims not in {4, 3}:
        raise ValueError("Input image must be 3D or 4D array.")
    
    if img_dims == 3:
        img = np.expand_dims(img, axis=0)
    
    _, _, h, w = img.shape
    
    if h == 512 and w == 512:
        if img_dims == 3:
            img = np.squeeze(img, axis=0)
        return img
    
    if h > 512 or w > 512:
        img = interpolate_to_target_width(img, target_size=512)
    
    img = pad_square(img, target_size=512)
    
    if img_dims == 3:
        img = np.squeeze(img, axis=0)
    
    return img


def resize_and_pad(
    img: np.ndarray, 
    target_size: Optional[int] = None, 
    fill_value: int = 0,
    channel_first: bool = True  # <--- Added explicit flag (Default True for PyTorch/Medical)
) -> np.ndarray:
    
    # 1. Standardize to OpenCV format (H, W, C)
    if channel_first:
        # Input is (C, H, W) -> Transpose to (H, W, C)
        # Handle case where input is (H, W) (ndim=2) by expanding dims first
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        img = img.transpose(1, 2, 0)
    
    # Now img is guaranteed to be (H, W, C) or (H, W)
    h, w = img.shape[:2]
    
    # 2. Logic Branch: Resize vs. Just Square
    if target_size is not None:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        is_binary = (len(np.unique(img)) <= 2)
        interpolation = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        final_size = target_size
    else:
        resized = img
        new_w, new_h = w, h
        final_size = max(h, w) # Target is the largest side
        
    # Handle dimension drop (if C=1, cv2 removes the dim)
    if img.ndim == 3 and resized.ndim == 2:
        resized = np.expand_dims(resized, axis=2)
        
    # 3. Pad to Square
    delta_w = final_size - new_w
    delta_h = final_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(
        resized, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=fill_value
    )
    
    # 4. Restore to Channel First (C, H, W)
    if channel_first:
        if padded.ndim == 3:
            padded = padded.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
        else:
            padded = np.expand_dims(padded, axis=0)
            
    return padded


def _apply_transform(x, kind):
    if kind == "linear":
        return x
    
    if kind == "log1p":
        return np.log1p(np.maximum(x, 0))
    
    raise ValueError(f"Unknown transform: {kind}")


# Normalize image volume based on modality-specific config
def percentile_normalization(
    img_volume: np.ndarray,
    modality: str,
    lo: float = None,
    hi: float = None,
    mask: np.ndarray = None,
    use_mask: bool = True,
) -> np.ndarray:
    """
    img_volume: 3D ndarray (D, H, W) or 4D ndarray (1, D, H, W)
    modality: key in MODALITY_CFG
    """
    # 1. Input Validation & Formatting
    if img_volume.ndim == 4:
        if img_volume.shape[0] != 1:
            raise ValueError("4D image volume must have shape (1, D, H, W).")
        
        img_volume = np.squeeze(img_volume, axis=0)
    
    if img_volume.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {img_volume.shape}")

    modality = modality.upper()
    
    if modality not in MODALITY_CFG:
        raise ValueError(f"Unknown modality: {modality}")

    # 2. IMMEDIATE CAST TO FLOAT
    img = img_volume.astype(np.float32)
    
    # 3. Apply Transform
    cfg = MODALITY_CFG[modality]
    img = _apply_transform(img, cfg["transform"])

    # 4. Calculate Statistics (Masked)
    if lo is None or hi is None:
        if use_mask:
            if mask is None:
                # Create mask from valid pixels in the transformed image
                # Note: For log1p, 0 maps to 0, so this is safe.
                mask_bool = img > 1e-6 
            else:
                # Ensure mask is boolean
                mask_bool = mask > 0
                
            fg_pixels = img[mask_bool]
        else:
            fg_pixels = img.flatten()

        # 5. Handle Edge Cases (Empty mask)
        if fg_pixels.size == 0:
            return np.zeros_like(img, dtype=np.float32)

        # 6. Compute Percentiles
        lo, hi = np.percentile(fg_pixels, cfg["p"])

    # 7. Normalize
    if hi > lo:
        img_norm = (np.clip(img, lo, hi) - lo) / (hi - lo)
    else:
        # If flat image (e.g., all 0s), return 0s
        return np.zeros_like(img, dtype=np.float32)

    # 8. (Optional) Re-apply mask to clean background
    # Depending on your pipeline, you might want to force background to exactly 0
    if use_mask and mask is not None:
        mask_bool = mask > 0
        img_norm[~mask_bool] = 0.0

    return img_norm.astype(np.float32), lo, hi

# Compute mean and std for normalization
def get_mean_and_std(img_volume: np.ndarray) -> tuple[float, float]:
    """Normalize image volume to zero mean and unit variance."""
    if img_volume.ndim not in {3, 4}:
        raise ValueError("Input image volume must be 3D or 4D array.")
    
    if img_volume.ndim == 4 and img_volume.shape[0] != 1:
        raise ValueError("4D image volume must have shape (1, D, H, W).")
    
    if img_volume.ndim == 4:
        img_volume = np.squeeze(img_volume, axis=0)
    
    # compute mean and std over non-zero regions
    mask = (img_volume != 0)
    mean = np.mean(img_volume[mask])
    std = np.std(img_volume[mask])
    
    return mean, std


class MedicalSDFEncoder:
    def __init__(self, normalize_per_instance: bool = True, max_dist_cap: float = None):
        """
        Args:
            normalize_per_instance (bool): 
                If True, scales the peak of *each channel* to 1.0 (255) independently.
                Useful for making small tumors as salient as large ones.
            max_dist_cap (float, optional):
                Only used if normalize_per_instance is False. 
                The distance value (in pixels) that corresponds to 255. 
                Example: 50.0 means any pixel >50px from boundary is clipped to 255.
        """
        self.normalize_per_instance = normalize_per_instance
        self.max_dist_cap = max_dist_cap if max_dist_cap else 50.0

    def mask_to_sdf(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Encodes Binary Mask -> Distance Map -> uint8 Image [0, 255].
        """
        # 1. Standardize Input to (C, H, W)
        if binary_mask.ndim == 2:
            mask_input = binary_mask[np.newaxis, ...]
        elif binary_mask.ndim == 3:
            mask_input = binary_mask
        else:
            raise ValueError(f"Input shape {binary_mask.shape} not supported.")

        C, H, W = mask_input.shape
        sdf_output = np.zeros((C, H, W), dtype=np.float32)

        # 2. Process Each Channel
        for c in range(C):
            channel_mask = mask_input[c]
            
            # distance_transform_edt computes distance to nearest BACKGROUND (0) pixel.
            # So inside the object = High values. Outside = 0.
            dist_map = distance_transform_edt(channel_mask)
            
            max_val = dist_map.max()
            
            if max_val > 0:
                if self.normalize_per_instance:
                    # Scale peak to 1.0 (Relative Distance)
                    sdf_output[c] = dist_map / max_val
                else:
                    # Scale by fixed cap (Absolute Distance)
                    # Use the cap to map [0, cap] -> [0.0, 1.0]
                    sdf_output[c] = dist_map / self.max_dist_cap
            
        # 3. Clip and Quantize to uint8
        # Any value > 1.0 (from the Absolute branch) gets clamped to 255
        sdf_output = np.round(np.clip(sdf_output, 0.0, 1.0) * 255.0).astype(np.uint8)

        # If input was 2D, return 2D (optional, but good for consistency)
        if binary_mask.ndim == 2:
            return sdf_output.squeeze(0)

        return sdf_output

    def sdf_to_mask(self, sdf_prediction: np.ndarray, threshold: int = 0) -> np.ndarray:
        """
        Decodes SDF uint8 Image [0, 255] -> Binary Mask [0, 255].
        
        Args:
            sdf_prediction: uint8 array.
            threshold: int (0-255). 
                       Use 0 if you want "any non-zero pixel".
                       Use 127 (50%) for robust cleanup of noisy model predictions.
        """
        # 1. Thresholding
        # Convert to boolean mask
        binary_mask_bool = (sdf_prediction > threshold)
        
        # 2. Scale to [0, 255] uint8
        binary_mask = (binary_mask_bool * 255).astype(np.uint8)
        
        return binary_mask


class MedicalSoftBinaryEncoder:
    def __init__(self, blur_radius=2.0):
        """
        Encoder to convert between hard binary masks (0/1) and soft VAE-friendly images.
        
        Args:
            blur_radius (float): The sigma for Gaussian Blur. 
                                 2.0 is recommended for 512x512 images.
        """
        self.blur_radius = blur_radius

    def mask_to_soft(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Converts a hard binary mask to a soft RGB uint8 image.
        
        Args:
            mask_array (np.ndarray): Input binary mask. Shape (H, W).
                                     Values should be 0 or 1.
        
        Returns:
            np.ndarray: Soft binary image. Shape (H, W, 3).
                        Values are uint8 [0, 255].
        """
        # 1. Validation and Setup
        mask_array = np.squeeze(mask_array) # Handle (1, H, W) or (H, W, 1)
        
        if mask_array.ndim != 2:
            raise ValueError(f"Input mask must be 2D, got shape {mask_array.shape}")
            
        # 2. Scale 0/1 -> 0/255
        mask_uint8 = (mask_array * 255).astype(np.uint8)
        
        # 3. Apply Gaussian Blur via PIL
        img = Image.fromarray(mask_uint8, mode='L')
        img_soft = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        return np.array(img_soft)

    def soft_to_mask(self, soft_array: np.ndarray) -> np.ndarray:
        """
        Converts a soft uint8 image back to a hard binary mask.
        
        Args:
            soft_array (np.ndarray): Soft input image. Shape (H, W, 3) or (H, W).
                                     Values are uint8 [0, 255].
                                     
        Returns:
            np.ndarray: Binary mask. Shape (H, W).
                        Values are uint8 [0, 255].
        """
        # 1. Handle RGB input (Average channels)
        if soft_array.ndim == 3 and soft_array.shape[2] == 3:
            grayscale = soft_array.mean(axis=2)
        elif soft_array.ndim == 3 and soft_array.shape[0] == 3:
            # Handle PyTorch style (C, H, W) just in case
            grayscale = soft_array.mean(axis=0)
        else:
            grayscale = soft_array

        # 2. Threshold at midpoint (127)
        # > 127 is foreground (1), <= 127 is background (0)
        binary_mask = (grayscale > 127)
        binary_mask = (binary_mask * 255).astype(np.uint8)
        
        return binary_mask
