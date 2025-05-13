import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2 # Be sure to use v2 import
from utils.camera import Camera
from dataset import MultiCamDataset, SetType
from utils.visual import plot
from utils.utils import masks_to_boxes
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
    # plot([(left_img, left_target), (left_img, new_left_target)], col_title=["Original", "Predicted"])

    # Initialize the FoundationStereo predictor
    fs_predictor = FoundationStereoPredictor(
        checkpoint_path="FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
        config_path="FoundationStereo/pretrained_models/23-51-11/cfg.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Get the 3D points from the FoundationStereo predictor
    K = firefly_left.camera_matrix
    baseline = np.abs(firefly_left.transforms["firefly_right"][1][0])
    depth_map = fs_predictor.predict_depth(
        left_img = left_img,
        right_img = right_img,
        focal_length = K[0, 0],
        baseline = baseline, 
        scale = 0.8,
        vis = True
    )

    # Collect the (u, v, z) for each mask in the firefly left image
    ffl_uvz = []
    for mask in left_masks:
        m = mask.cpu().numpy().astype(bool)
        vs, us = np.nonzero(m)
        zs = depth_map[vs, us]
        uvz = np.stack([us, vs, zs], axis=1) # shape (N,3)
        ffl_uvz.append(uvz)

    # Unproject the pixels from the firefly left image to 3D points in firefly left frame
    K_inv = np.linalg.inv(K)
    ffl_pts3d = []
    for uvz in ffl_uvz:
        us, vs, zs = uvz[:, 0], uvz[:, 1], uvz[:, 2]
        homog = np.stack([us, vs, np.ones_like(us)], axis=1) # (N,3)
        rays = (K_inv @ homog.T).T # (N,3)
        pts3d = rays * zs[:, np.newaxis] # [x/z, y/z, 1] * z = [x, y, z]
        ffl_pts3d.append(pts3d)

    # Transform the points to the ximea frame
    R, t = firefly_left.transforms["ximea"]
    xim_pts3d = []
    for pts in ffl_pts3d:
        xim = (R @ pts.T + t.reshape(3, 1)).T # (N,3)
        xim_pts3d.append(xim)

    # Project into the ximea image
    K2 = ximea.camera_matrix
    projected = []
    for pts in xim_pts3d:
        uv2_h = (K2 @ pts.T).T # (N,3) homogeneous
        uv2   = uv2_h[:, :2] / uv2_h[:, [2]] # divide by z
        projected.append(uv2)

    # Create the masks for the ximea image
    ximea_masks = []
    H, W = ximea_img[0].shape
    for uv2 in projected:
        us = torch.tensor(uv2[:, 0].astype(int), dtype=torch.long)
        vs = torch.tensor(uv2[:, 1].astype(int), dtype=torch.long)

        # Keep only pixels inside [0..W-1] and [0..H-1]
        valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us, vs = us[valid], vs[valid]

        # Create a mask for the ximea image
        mask = torch.zeros((H, W), dtype=torch.uint8)
        mask[vs, us] = True
        ximea_masks.append(mask)
    ximea_masks = torch.stack(ximea_masks, dim=0)
    
    # Convert the masks to boxes
    ximea_boxes = masks_to_boxes(ximea_masks)
    
    # Create the target for the ximea image
    new_ximea_target = ximea_target.copy()
    new_ximea_target['masks'] = ximea_masks
    new_ximea_target['boxes'] = ximea_boxes
    new_ximea_target['labels'] = left_target['labels']

    # Plot the original and predicted masks
    plot([(left_img, new_left_target), (ximea_img, new_ximea_target)], col_title=["Firefly", "Ximea"])








if __name__ == '__main__':
    demo()
