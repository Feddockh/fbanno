import os
from PIL import Image
import torch
from torchvision.transforms import v2 # Be sure to use v2 import
from utils.camera import Camera
from dataset import MultiCamDataset, SetType
from utils.visual import plot
from predictors import predict_sam_masks


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")

def demo():
    # Create the cameras
    cam0 = Camera("firefly_left")
    cam1 = Camera("firefly_right")
    cam2 = Camera("ximea")
    cameras = [cam0, cam1, cam2]

    # Load the camera parameters
    for cam in cameras:
        cam.load_params()
        cam.compute_maps()

    # Define the transforms
    transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        # v2.RandomHorizontalFlip(p=0.5),
    ])

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.ALL, transforms=transforms, undistort_rectify=True)
    
    # Sample an image from the dataset
    view_idx = 100
    img, target, img_path = dataset[view_idx][cam0.name]

    # Pass the image and target to the SAM2 predictor
    masks = predict_sam_masks(img, target['boxes'])
    new_target = target.copy()
    new_target['masks'] = masks
    plot([(img, target), (img, new_target)], col_title=["Original", "Predicted"])

if __name__ == '__main__':
    demo()
