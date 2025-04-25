import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List
from utils.utils import image_to_numpy, bbox_to_numpy


def predict_sam_masks(image: torch.Tensor | np.ndarray | Image.Image,
                     boxes: torch.Tensor | np.ndarray | List,
                     model_type: str = "facebook/sam2-hiera-tiny",
                     checkpoint: str = None, config: str = None,
                     device: str = None) -> torch.Tensor:
    """
    Given an image and a bbox, returns the highest-scoring SAM2 mask as a torch.Tensor.

    Args:
        image: PIL.Image, np.ndarray (H×W×C uint8), or torch.Tensor
                    (either C×H×W or H×W×C, float [0,1] or uint8 [0,255]).
        boxes: (4,) array-like / Tensor OR (N,4) array-like / Tensor of [x1,y1,x2,y2].
        model_type: pretrained SAM2 model identifier (hf).
        checkpoint: path to a custom SAM2 .pth checkpoint (optional).
        config: path to SAM2 config.yaml if using a custom checkpoint.
        device: torch.device; defaults to cuda if available.

    Returns:
        best_mask: torch.BoolTensor of shape (N, H, W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build or load SAM predictor
    if checkpoint and config:
        from sam2.build_sam import build_sam2
        sam_model = build_sam2(config, checkpoint).to(device)
        predictor = SAM2ImagePredictor(sam_model)
    else:
        predictor = SAM2ImagePredictor.from_pretrained(model_type)
        predictor.model.to(device)
    predictor.model.eval()

    # Convert the image to a numpy array
    img_np = image_to_numpy(image)

    # Set the image in the predictor
    predictor.set_image(img_np)

    # Convert the bbox to a numpy array
    bbox_np = bbox_to_numpy(boxes)

    # Get the mask prediction for each box
    if bbox_np.ndim == 1:
        bbox_np = bbox_np.reshape(1, 4)
    
    # Iterate over each box and get the predicted masks
    masks = []
    for i in range(bbox_np.shape[0]):
        # Get the mask for the current box
        mask, scores, _ = predictor.predict(box=bbox_np[i])

        # Pick the highest-scoring mask
        best_idx  = int(np.argmax(scores))
        best_mask = mask[best_idx]
        
        # Remove mask outside the box
        x1, y1, x2, y2 = map(int, bbox_np[i])
        bbox_mask = np.zeros_like(best_mask, dtype=bool)
        bbox_mask[y1:y2, x1:x2] = best_mask[y1:y2, x1:x2]
        
        # Append the mask to the list
        masks.append(bbox_mask)

    # Stack the list of masks into a 3D array
    masks = np.stack(masks, axis=0)

    # Convert the mask to a torch tensor and return
    t_masks = torch.tensor(masks, dtype=torch.uint8)
    return t_masks

