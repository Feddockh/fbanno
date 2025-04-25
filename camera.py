import os
import cv2
import yaml
import numpy as np
from typing import List, Tuple, Dict


EXTENSION = ".png"
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_files")

class Camera:
    def __init__(self, name: str):
        self.name = name
        self.width: int = 0
        self.height: int = 0
        self.camera_matrix: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        self.dist_coeffs: np.ndarray = np.zeros((5,), dtype=np.float32)
        self.rectification_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.projection_matrix: np.ndarray = np.zeros((3, 4), dtype=np.float32)

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

# This class may be removed in the future
class MultiCamCapture:
    def __init__(self, base_dir: str, cameras: List[Camera], id: str):
        """
        Store the paths to the images for each camera used in a single capture.
        """
        self.base_dir = base_dir
        self.cameras = cameras
        self.id = id
        self.image_paths: Dict[str, str] = {
            cam.name: os.path.join(self.base_dir, cam.name, id + EXTENSION) for cam in cameras
        }
        self.images: Dict[str, np.ndarray] = {}
        
    def load_images(self) -> Dict[str, np.ndarray]:
        """
        Load the images from the image paths.
        """
        for cam_name, path in self.image_paths.items():
            self.images[cam_name] = cv2.imread(path)
        return self.images
    
    def unload_images(self):
        """
        Unload the images to free up memory.
        """
        self.images = {}
    
    def get_image(self, camera_name: str) -> np.ndarray:
        """
        Get the image for a given camera.
        """
        if camera_name not in self.images:
            raise ValueError(f"Camera {camera_name} not found in MultiCamCapture.")
        if self.images[camera_name] is None:
            raise ValueError(f"Image for camera {camera_name} is not loaded.")
        return self.images[camera_name]
    
    def undistort_rectify_images(self):
        """
        Rectify the images for each camera.
        """
        for cam in self.cameras:
            
            # Create undistort/rectify map
            map1, map2 = cv2.initUndistortRectifyMap(
                cam.camera_matrix, 
                cam.dist_coeffs, 
                cam.rectification_matrix, 
                cam.projection_matrix, 
                (cam.width, cam.height), 
                cv2.CV_16SC2
            )

            # Rectify the image using the computed maps
            img = self.images[cam.name]
            if img is None:
                print(f"Warning: No image found for camera {cam.name}.")
                continue
            if img.shape[1] != cam.width or img.shape[0] != cam.height:
                print(f"Warning: Image size {img.shape} does not match camera parameters for {cam.name}.")
                continue
            rectified_image = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            self.images[cam.name] = rectified_image

        return self.images
