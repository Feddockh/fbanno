import os
from PIL import Image
import torch
from torchvision.transforms import v2 # Be sure to use v2 import
from utils.camera import Camera
from dataset import MultiCamDataset, SetType
from utils.visual import plot
from predictors import predict_sam_masks, FoundationStereoPredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")

def demo():
    # Create the cameras
    firefly_left = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea = Camera("ximea")
    cameras = [firefly_left, firefly_right, ximea]

    # Load the camera parameters
    for cam in cameras:
        cam.load_params()
        cam.compute_maps()

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.ALL, undistort_rectify=True)
    
    # Sample an image from the dataset
    view_idx = 100
    left_img, left_target, _ = dataset[view_idx][firefly_left.name]
    right_img, right_target, _ = dataset[view_idx][firefly_right.name]
    ximea_img, ximea_target, _ = dataset[view_idx][ximea.name]

    # Pass the image and target to the SAM2 predictor
    left_masks = predict_sam_masks(left_img, left_target['boxes'])
    new_left_target = left_target.copy()
    new_left_target['masks'] = left_masks
    plot([(left_img, left_target), (left_img, new_left_target)], col_title=["Original", "Predicted"])

    # Initialize the FoundationStereo predictor
    fs_predictor = FoundationStereoPredictor(
        checkpoint_path="FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
        config_path="FoundationStereo/pretrained_models/23-51-11/cfg.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Get the 3D points from the FoundationStereo predictor
    depth_map = fs_predictor.predict_depth(
        left_img=left_img,
        right_img=right_img,
        focal_length=firefly_left.camera_matrix[0, 0],
        baseline=firefly_right.projection_matrix[0, 3] / firefly_right.projection_matrix[0, 0],
        scale=0.8,
        vis=True
    )

    # Get the 3D masks using the depth map and the left masks
    # left_masks

    # Go from 3D points in firefly left to 3D points in ximea
    # R, t = firefly_left.transforms["ximea"]
    
    # Show the 3D points in the ximea coordinate system




if __name__ == '__main__':
    demo()
