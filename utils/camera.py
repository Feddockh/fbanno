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
        self.inv_map1: np.ndarray = None
        self.inv_map2: np.ndarray = None

    def __eq__(self, value):
        """
        Check if two Camera objects are equal based on their name.
        """
        if isinstance(value, Camera):
            return self.name == value.name
        return False

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
        if self.width == 0 or self.height == 0:
            raise ValueError("Camera parameters not loaded properly. Be sure to call load_params() first.")

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, 
            self.dist_coeffs, 
            self.rectification_matrix, 
            self.projection_matrix, 
            (self.width, self.height), 
            cv2.CV_16SC2
        )

    def compute_inverse_maps(self):
        if self.map1 is None or self.map2 is None:
            self.compute_maps()

        # Turn your fixed-point maps into a single float32 2‑channel map:
        #    map_fwd is H×W×2 float32 where map_fwd[v,u] = (x_src, y_src)
        map_fwd, _ = cv2.convertMaps(self.map1, self.map2, cv2.CV_32FC2)

        H, W = map_fwd.shape[:2]
        w0, h0 = self.width, self.height

        # Allocate empty inverse map and mark all as unfilled
        inv_map = np.full((h0, w0, 2), np.nan, dtype=np.float32)

        # For every pixel in the RECTIFIED frame, see where it came from
        for v in range(H):
            for u in range(W):
                x_src, y_src = map_fwd[v, u]
                ix, iy = int(round(x_src)), int(round(y_src))
                if 0 <= ix < w0 and 0 <= iy < h0:
                    # At raw‐image pixel (ix,iy), store that it came from (u,v)
                    inv_map[iy, ix, 0] = u
                    inv_map[iy, ix, 1] = v

        # Fill any holes (pixels never visited) by inpainting each channel separately
        mask = np.isnan(inv_map[..., 0]).astype(np.uint8)
        # OpenCV inpaint requires 8‑bit 1‑channel mask and float image
        inv_map[..., 0] = cv2.inpaint(inv_map[..., 0], mask, 3, cv2.INPAINT_NS)
        inv_map[..., 1] = cv2.inpaint(inv_map[..., 1], mask, 3, cv2.INPAINT_NS)

        self.inv_map1 = inv_map
        self.inv_map2 = None

    def undistort_rectify_image(self, image: np.ndarray | torch.Tensor | PIL_Image.Image, inverse: bool = False) \
            -> np.ndarray | torch.Tensor | PIL_Image.Image:
        """
        Undistort and rectify the image using the camera parameters.

        Args:
            image: np.ndarray H×W×C | torch.Tensor C×H×W | PIL.Image.
            inverse: If True, undistort and rectify the image to the original image space.
            
        Returns:
            rectified_image: Undistorted and rectified image in the same format as input.
        """
        # Compute undistort/rectify map if not already computed
        if self.map1 is None or self.map2 is None:
            self.compute_maps()
        if inverse and (self.inv_map1 is None or self.inv_map2 is None):
            self.compute_inverse_maps()

        # Use the appropriate map based on the inverse flag
        if not inverse:
            map1, map2 = self.map1, self.map2
        else:
            map1, map2 = self.inv_map1, self.inv_map2

        # Rectify the image using the computed maps
        img_np = image_to_numpy(image) # (H,W,C) uint8
        # print(f"Image shape: {img_np.shape}")

        # # Display the original image
        # import matplotlib.pyplot as plt
        # plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        # plt.title("Numpy Image")
        # plt.axis("off")
        # plt.show()

        rectified_image = cv2.remap(img_np, map1, map2, cv2.INTER_LINEAR)
        rectified_image = np.atleast_3d(rectified_image)
        # print(f"Rectified image shape: {rectified_image.shape}")

        # # Display the rectified image
        # plt.imshow(cv2.cvtColor(rectified_image, cv2.COLOR_BGR2RGB))
        # plt.title("Rectified Image")
        # plt.axis("off")
        # plt.show()
        
        # Convert back to original format
        if isinstance(image, torch.Tensor):
            # Convert to uint8 and permute to (C,H,W)
            rectified_image = torch.tensor(rectified_image, dtype=torch.uint8).permute(2, 0, 1) # (H,W,C) --> (C,H,W)
        elif isinstance(image, PIL_Image.Image):
            rectified_image = PIL_Image.fromarray(rectified_image)
        else:
            rectified_image = np.array(rectified_image, dtype=np.uint8)

        return rectified_image

    def undistort_rectify_target(self, target: Dict[str, torch.Tensor], inverse: bool = False) \
            -> Dict[str, torch.Tensor]:
        """
        Undistort and rectify the image using the camera parameters.

        Args:
            target: Dictionary containing the target annotations
                - 'boxes': BoundingBoxes of shape (N, 4) with bounding boxes in [x1, y1, x2, y2] format.
                - 'masks': Masks of shape (N, H, W) with binary masks.
                - 'labels': torch.Tensor of shape (N,) with class labels.
            inverse: If True, undistort and rectify the target to the original image space.

        Returns:
            target: Undistorted and rectified target.
        """
        # Compute undistort/rectify map if not already computed
        if self.map1 is None or self.map2 is None:
            self.compute_maps()
        if inverse and (self.inv_map1 is None or self.inv_map2 is None):
            self.compute_inverse_maps()

        # Use the appropriate map based on the inverse flag
        if not inverse:
            map1, map2 = self.map1, self.map2
        else:
            map1, map2 = self.inv_map1, self.inv_map2
        
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
                bbox_mask = cv2.remap(bbox_mask, map1, map2, cv2.INTER_NEAREST)
                
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
                mask = cv2.remap(masks_np[i], map1, map2, cv2.INTER_NEAREST)
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
        target_new = target.copy()
        target_new['boxes'] = boxes_tv
        target_new['masks'] = masks_tv

        return target_new
    
    def copy(self) -> "Camera":
        """
        Create a deep copy of the Camera object including all its parameters.
        """
        new_cam = Camera(self.name)
        new_cam.width = self.width
        new_cam.height = self.height
        new_cam.camera_matrix = self.camera_matrix.copy()
        new_cam.dist_coeffs = self.dist_coeffs.copy()
        new_cam.rectification_matrix = self.rectification_matrix.copy()
        new_cam.projection_matrix = self.projection_matrix.copy()
        new_cam.transforms = {k: (v[0].copy(), v[1].copy()) for k, v in self.transforms.items()}
        new_cam.map1 = self.map1.copy() if self.map1 is not None else None
        new_cam.map2 = self.map2.copy() if self.map2 is not None else None
        new_cam.inv_map1 = self.inv_map1.copy() if self.inv_map1 is not None else None
        new_cam.inv_map2 = self.inv_map2.copy() if self.inv_map2 is not None else None
        return new_cam


