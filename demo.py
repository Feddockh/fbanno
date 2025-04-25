import os
from PIL import Image
import torch
from torchvision.transforms import v2 # Be sure to use v2 import
from camera import Camera
from dataset import MultiCamDataset, SetType
from utils.visual import plot
from predictors import predict_sam_masks


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "rivendale_dataset")

def demo():
    # Create the cameras
    cam0 = Camera("firefly_left") # Use this for the rivendale dataset
    # cam0 = Camera("cam0") # Use this for the erwiam dataset
    cameras = [cam0]

    # Define the transforms
    transforms = v2.Compose([
        v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.ALL, transforms=transforms)
    view_idx = 100
    img, target, img_path = dataset[view_idx][cam0.name]

    # Clear the target masks
    labels = target['labels']
    target['masks'] = torch.zeros((len(labels), img.shape[1], img.shape[2]))

    # Print the shape of the image and annotations and plot the image with annotations
    print(f"Image shape: {img.shape}")
    print(f"Annotation boxes shape: {target['boxes'].shape}")
    print(f"Annotation masks shape: {target['masks'].shape}")
    print(f"Annotation labels shape: {target['labels'].shape}")
    print(f"Annotation labels: {target['labels']}")
    plot([(img, target)])

    # Pass the image and target to the SAM2 predictor
    masks = predict_sam_masks(img, target['boxes'])
    target['masks'] = masks
    print(f"Predicted mask shape: {target['masks'].shape}")
    plot([(img, target)])

if __name__ == '__main__':
    demo()
