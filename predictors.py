import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List, Optional, Union
from utils.utils import image_to_numpy, bbox_to_numpy
from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import vis_disparity, depth2xyzmap
from utils.visual import plot


class SAM2Predictor:
    def __init__(self, model_type: str = "facebook/sam2-hiera-tiny",
        checkpoint: Optional[str] = None, config: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None):
        """
        Wraps SAM2ImagePredictor so it’s instantiated only once.

        Args:
            model_type: HF model identifier (e.g. "facebook/sam2-hiera-tiny").
            checkpoint: path to custom .pth checkpoint (optional).
            config:     path to custom SAM2 config.yaml (optional).
            device:     "cuda"/"cpu" or torch.device (defaults to cuda if available).
        """
        # pick device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # build the SAM model
        if checkpoint and config:
            from sam2.build_sam import build_sam2
            sam_model = build_sam2(config, checkpoint).to(self.device)
            self.predictor = SAM2ImagePredictor(sam_model)
        else:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_type)
            self.predictor.model.to(self.device)

        self.predictor.model.eval()

    def predict(self, image: Union[torch.Tensor, np.ndarray, Image.Image],
        boxes: Union[torch.Tensor, np.ndarray, List], crop: bool = True) -> torch.Tensor:
        """
        Given an image and N box(es), returns an (N, H, W) uint8 mask tensor.

        Args:
            image: PIL.Image, numpy H×W×C uint8, or torch.Tensor (C×H×W or H×W×C).
            boxes: (4,) or (N,4) array-like of [x1,y1,x2,y2].

        Returns:
            masks: torch.uint8 tensor of shape (N, H, W), values in {0,1}.
        """
        # 1) Prepare image & model
        img_np = image_to_numpy(image)

        # Expand to 3 channels if grayscale
        if img_np.ndim == 2:
            img_np = img_np[:, :, None]
        
        # Repeat channels if needed
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        self.predictor.set_image(img_np)

        # 2) Prepare boxes
        bbox_np = bbox_to_numpy(boxes)
        if bbox_np.ndim == 1:
            bbox_np = bbox_np[None, :]

        # 3) Predict per-box and pick best mask
        masks = []
        for box in bbox_np:
            masks_i, scores, _ = self.predictor.predict(box=box)

            # Sort indices by descending score
            # idxs = np.argsort(scores)[::-1][:2]
            # mask = np.sum(masks_i[idxs], axis=0) > 0

            best = int(np.argmax(scores))
            mask = masks_i[best]

            # zero-out outside the box
            if crop:
                x1, y1, x2, y2 = map(int, box)
                cropped = np.zeros_like(mask)
                cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                masks.append(cropped)
            else:
                masks.append(mask)

        if len(masks) == 0:
            raise ValueError("No masks found. Check the input image and boxes.")

        # 4) return as torch tensor
        masks = np.stack(masks, axis=0)
        return torch.from_numpy(masks.astype(np.uint8))

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
        


