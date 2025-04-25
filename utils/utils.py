import torch
import numpy as np
from PIL import Image
from typing import List, Tuple


def image_to_numpy(image: torch.Tensor | np.ndarray | Image.Image) -> np.ndarray:
    """
    Convert a torch.Tensor, np.ndarray, or PIL.Image to a numpy array.

    Args:
        image: PIL.Image, np.ndarray (H×W×C uint8), or torch.Tensor
                    (either C×H×W or H×W×C, float [0,1] or uint8 [0,255]).

    Returns:
        img_np: numpy array of shape (H, W, C) with dtype uint8.
    """
    # Torch tensor has shape [C,H,W], we need to convert it to [H,W,C] for numpy
    if isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        # If the tensor is 2D, add a channel dimension
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0) # [H,W] --> [1,H,W]
        # If the tensor is 3D, permute it to [H,W,C]
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            img_t = img_t.permute(1, 2, 0) # [C,H,W] --> [H,W,C]
        # Scale the image to [0,255] and convert to uint8 if it is a float tensor
        if torch.is_floating_point(img_t):
            img_t = (img_t * 255.0).clamp(0, 255).to(torch.uint8)
        # If not a float tensor, convert to uint8
        else:
            img_t = img_t.to(torch.uint8)
        img_np = img_t.numpy()
    # If the image is already a numpy array, do nothing
    elif isinstance(image, np.ndarray):
        img_np = image
    # If the image is a PIL Image, convert it to a numpy array
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return img_np

def bbox_to_numpy(bbox: torch.Tensor | np.ndarray | List) -> np.ndarray:
    """
    Convert a torch.Tensor, np.ndarray, or list to a numpy array.

    Args:
        bbox: (4,) array-like / Tensor OR (N,4) array-like / Tensor of [x1,y1,x2,y2].

    Returns:
        bbox_np: numpy array of shape (N, 4) with dtype float.
    """
    if isinstance(bbox, torch.Tensor):
        bbox_np = bbox.detach().cpu().numpy().astype(float)
    else:
        bbox_np = np.array(bbox, dtype=float)

    return bbox_np

def masks_to_numpy(masks: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a torch.Tensor or np.ndarray of masks to a numpy array.

    Args:
        masks: (N, H, W) tensor of masks.

    Returns:
        masks_np: numpy array of shape (N, H, W) with dtype uint8.
    """
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    else:
        masks_np = np.array(masks, dtype=np.uint8)

    return masks_np

def masks_to_boxes(masks: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Convert a tensor of masks to bounding boxes.

    Args:
        masks: (N, H, W) tensor of masks.

    Returns:
        boxes: (N, 4) tensor of bounding boxes in [x1, y1, x2, y2] format.
    """
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.array(masks)

    boxes = []
    for mask in masks_np:
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            boxes.append([0, 0, 0, 0])
            continue
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()
        boxes.append([x1, y1, x2, y2])

    return torch.tensor(boxes).float()