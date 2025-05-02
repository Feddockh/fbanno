import os
import cv2
import yaml
import numpy as np
import PIL.Image as PIL_Image
import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask
from typing import List, Tuple, Dict
from utils.utils import image_to_numpy, bbox_to_numpy, masks_to_numpy, masks_to_boxes


EXTENSION = ".png"
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_files")

class Camera:
    def __init__(self, name: str):
        self.name = name
        self.width: int = 0
        self.height: int = 0
        self.camera_matrix: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        self.dist_coeffs: np.ndarray = np.zeros((5,), dtype=np.float32)
        self.rectification_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.projection_matrix: np.ndarray = np.zeros((3, 4), dtype=np.float32)
        self.transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self.map1: np.ndarray = None
        self.map2: np.ndarray = None

    def load_params(self):
        """
        Load the camera parameters from a YAML file.
        """
        yaml_file = os.path.join(CALIBRATION_DIR, f"{self.name}.yaml")
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Calibration file {yaml_file} not found.")
        with open(yaml_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        self.width = int(calib_data['image_width'])
        self.height = int(calib_data['image_height'])
        self.camera_matrix = np.array(calib_data['camera_matrix']['data'], dtype=np.float32).reshape((3, 3))
        self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float32)
        self.rectification_matrix = np.array(calib_data['rectification_matrix']['data'], dtype=np.float32).reshape((3, 3))
        self.projection_matrix = np.array(calib_data['projection_matrix']['data'], dtype=np.float32).reshape((3, 4))
        # Load transforms if available
        if 'transforms' in calib_data:
            for name, transform in calib_data['transforms'].items():
                R = np.array(transform['R'], dtype=np.float32).reshape((3, 3))
                t = np.array(transform['t'], dtype=np.float32).reshape((3,))
                self.transforms[name] = (R, t)

    def compute_maps(self):
        """
        Compute the undistort and rectify maps for the camera.
        """
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, 
            self.dist_coeffs, 
            self.rectification_matrix, 
            self.projection_matrix, 
            (self.width, self.height), 
            cv2.CV_16SC2
        )

    def undistort_rectify_image(self, image: np.ndarray | torch.Tensor | PIL_Image.Image) -> np.ndarray | torch.Tensor | PIL_Image.Image:
        """
        Undistort and rectify the image using the camera parameters.

        Args:
            image: np.ndarray H×W×C | torch.Tensor C×H×W | PIL.Image.
            
        Returns:
            rectified_image: Undistorted and rectified image in the same format as input.
        """
        # Compute undistort/rectify map if not already computed
        if self.map1 is None or self.map2 is None:
            self.compute_maps()

        # Rectify the image using the computed maps
        img_np = image_to_numpy(image) # [H,W,C] uint8
        rectified_image = cv2.remap(img_np, self.map1, self.map2, cv2.INTER_LINEAR)
        
        # Promote the rectified image to at least 3 dims
        if len(rectified_image.shape) == 2:
            rectified_image = rectified_image[:, :, np.newaxis] # [H,W] --> [H,W,1]

        # Comvert back to original format
        if isinstance(image, torch.Tensor):
            # Convert to uint8 and permute to [C,H,W]
            rectified_image = torch.tensor(rectified_image, dtype=torch.uint8).permute(2, 0, 1) # [H,W,C] --> [C,H,W]
        elif isinstance(image, PIL_Image.Image):
            rectified_image = PIL_Image.fromarray(rectified_image)
        else:
            rectified_image = np.array(rectified_image, dtype=np.uint8)

        return rectified_image

    def undistort_rectify_target(self, target: Dict[str, torch.Tensor]) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Undistort and rectify the image using the camera parameters.

        Args:
            target: Dictionary containing the target annotations
                - 'boxes': BoundingBoxes of shape (N, 4) with bounding boxes in [x1, y1, x2, y2] format.
                - 'masks': Masks of shape (N, H, W) with binary masks.
                - 'labels': torch.Tensor of shape (N,) with class labels.

        Returns:
            image or (image, target): Undistorted and rectified image.
        """
        # Compute undistort/rectify map if not already computed
        if self.map1 is None or self.map2 is None:
            self.compute_maps()
        
        # Get the bounding boxes from the target
        boxes = target['boxes']
        if boxes.numel() > 0:

            # Convert the boxes to numpy array
            boxes_np = bbox_to_numpy(boxes)

            # Undistort and rectify each bounding box
            boxes = []
            for i in range(boxes_np.shape[0]):
                # Create a mask for each bounding box
                bbox_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, boxes_np[i])
                bbox_mask[y1:y2, x1:x2] = 1

                # Perform the undistortion and rectification on the mask
                bbox_mask = cv2.remap(bbox_mask, self.map1, self.map2, cv2.INTER_LINEAR)
                
                # Convert the mask back to a bounding box by getting the max and min coordinates
                box = masks_to_boxes(bbox_mask.reshape(1, self.height, self.width))[0]
                boxes.append(box)
            
            # Convert the list of boxes to a stack
            boxes = np.stack(boxes, axis=0)

            # Convert the boxes to a tensor
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

            # Convert the boxes to tv tensor format
            canvas_size = (self.height, self.width)
            boxes_tv = BoundingBoxes(boxes_tensor, format=BoundingBoxFormat.XYXY, canvas_size=canvas_size)

        # If no boxes are provided, pass the original boxes (empty tensor)
        else:
            boxes_tv = target['boxes']

        # Get the masks from the target
        masks = target['masks']
        if masks.numel() > 0 and torch.sum(masks) > 0:

            # Convert the masks to numpy array
            masks_np = masks_to_numpy(masks)

            # Create a mask for each bounding box
            masks = []
            for i in range(masks_np.shape[0]):

                # Perform the undistortion and rectification on the mask
                mask = cv2.remap(masks_np[i], self.map1, self.map2, cv2.INTER_LINEAR)
                masks.append(mask)

            # Convert the list of masks to a stack
            masks = np.stack(masks, axis=0)

            # Convert the masks to a tensor
            masks_tensor = torch.tensor(masks, dtype=torch.uint8)
            
            # Convert the masks to tv tensor format
            masks_tv = Mask(masks_tensor)

        # If no masks are provided, pass the original masks (may be empty or just zeros)
        else:
            masks_tv = target['masks']

        # Assemble the target dictionary
        target_new = {}
        target_new['boxes'] = boxes_tv
        target_new['masks'] = masks_tv
        target_new['labels'] = target['labels']

        return target_new
