import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List
from utils.utils import image_to_numpy, bbox_to_numpy
from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import vis_disparity, depth2xyzmap

from utils.visual import plot


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

class FoundationStereoPredictor:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = None):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = OmegaConf.load(self.config_path)
        if 'vit_size' not in args:
            args['vit_size'] = 'vitl'
        self.model = FoundationStereo(args)
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def predict_depth(self, left_img: torch.Tensor, right_img: torch.Tensor, focal_length: float, 
                          baseline: float, scale: float = 1.0, vis: bool = False) -> torch.Tensor:
        """
        Predicts depth map from stereo images using Foundation Stereo.

        Args:
            left_img (Torch.tensor): Left image [C×H×W].
            right_img (Torch.tensor): Right image [C×H×W].
            focal_length (float): Focal length of the camera.
            baseline (float): Baseline distance between the two cameras.
            scale: Scale factor for the images (default is 1.0).
            vis: Whether to visualize the disparity map (default is False).

        Returns:
            depth_map (Torch.tensor): Depth map [H×W].
        """
        C, H, W = left_img.shape

        # Convert to [1, C, H, W] and float
        left_img = left_img.to(self.device).float().unsqueeze(0)
        right_img = right_img.to(self.device).float().unsqueeze(0)

        # Scale the images
        if scale < 1:
            left_img = F.interpolate(left_img, scale_factor=scale, mode='bilinear')
            right_img = F.interpolate(right_img, scale_factor=scale, mode='bilinear')

        # Pad the images to the nearest multiple of 32
        padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
        left_img_pad, right_img_pad = padder.pad(left_img, right_img)

        with torch.no_grad():
            disparity_map = self.model.forward(left_img_pad, right_img_pad, iters=32, test_mode=True)
        
        # Unpad the disparity map
        disparity_map = padder.unpad(disparity_map.float())

        # Scale the disparity map to the original size
        if scale < 1:
            disparity_map = F.interpolate(disparity_map, size=(H, W), mode='bilinear')

        # Reshape the disparity map to the original image size
        disparity_map = disparity_map.data.cpu().numpy().reshape(H, W)
        
        if vis:
            disp_vis = vis_disparity(disparity_map)
            plot([left_img.squeeze(), disp_vis], col_title=["Left Image", "Disparity Map"])

        # Compute the depth map
        depth_map = (focal_length * scale * baseline) / disparity_map

        if vis:
            depth_vis = np.abs(depth_map) * 255.0 / np.max(np.abs(depth_map))
            plot([left_img.squeeze(), depth_vis], col_title=["Left Image", "Depth Map"], cmap="RdBu")

        return depth_map
        


