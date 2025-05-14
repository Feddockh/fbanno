import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2 # Be sure to use v2 import
import open3d as o3d
from utils.camera import Camera
from dataset import MultiCamDataset, SetType, save_annotations_coco
from utils.visual import plot
from utils.utils import masks_to_boxes
from predictors import SAM2Predictor, FoundationStereoPredictor
from components import sample_frame, unproject_masks_to_3d, radius_outlier_filter, \
    transform_frame, project_to_image


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rivendale_dataset")

def demo(idx=100):
    # Create the cameras
    firefly_left, firefly_right, ximea = Camera("firefly_left"), Camera("firefly_right"), Camera("ximea")
    cams = [firefly_left, firefly_right, ximea]
    for c in cams:
        c.load_params()
        c.compute_maps()

    # Initialize the SAM2 predictor
    sam_predictor = SAM2Predictor("facebook/sam2-hiera-tiny")

    # Initialize the FoundationStereo predictor
    fs_predictor = FoundationStereoPredictor(
        checkpoint_path="FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
        config_path="FoundationStereo/pretrained_models/23-51-11/cfg.yaml"
    )

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cams, set_type=SetType.ALL, undistort_rectify=True)

    # Sample an image from the dataset
    (l_img, l_target, _), (r_img, _, _), (x_img, x_target, _) = sample_frame(dataset, idx, cams)

    # Pass the image and target to the SAM2 predictor
    l_masks = sam_predictor.predict(l_img, l_target["boxes"])
    l_target_new = l_target.copy()
    l_target_new['masks'] = l_masks
    plot([(l_img, l_target), (l_img, l_target_new)], col_title=["Original Target", "SAM Predicted Target"])
    
    # Get the depth map from the FoundationStereo predictor
    K = firefly_left.camera_matrix
    b = np.abs(firefly_left.transforms["firefly_right"][1][0])
    depth_map = fs_predictor.predict_depth(l_img, r_img, focal_length=K[0,0], baseline=b, scale=0.8, vis=False)

    # Get the masked list of points in 3D using the depth map
    pts3d = unproject_masks_to_3d(l_masks, depth_map, K)

    # Filter the points using the radius outlier filter
    pts3d_filtered = radius_outlier_filter(pts3d, radius=0.01, min_neighbors=100)

    # Transform the points to the ximea frame
    R, t = firefly_left.transforms["ximea"]
    x_pts3d = transform_frame(pts3d_filtered, R, t)

    # Project into the ximea image
    K2 = ximea.camera_matrix
    uv2s = project_to_image(x_pts3d, K2)

    # Create the masks for the ximea image
    ximea_masks = []
    ximea_labels = []
    coverage_threshold = 0.9
    H, W = x_img[0].shape
    for i, uv2 in enumerate(uv2s):
        us = torch.tensor(uv2[:, 0].astype(int), dtype=torch.long)
        vs = torch.tensor(uv2[:, 1].astype(int), dtype=torch.long)

        # Compute the coverage of the mask
        inside = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        coverage = inside.sum().item() / us.shape[0]
        if coverage < coverage_threshold:
            continue

        # Create a mask for the ximea image
        mask = torch.zeros((H, W), dtype=torch.uint8)
        mask[vs, us] = True
        ximea_masks.append(mask)
        ximea_labels.append(l_target_new['labels'][i])
    
    # Convert the masks and labels to tensors
    # If no masks were created, create empty tensors
    if not ximea_masks:
        ximea_masks = torch.empty((0,H,W), dtype=torch.uint8)
        ximea_labels = torch.empty((0,), dtype=torch.int64)
        ximea_boxes = torch.empty((0,4))
    else:
        ximea_masks = torch.stack(ximea_masks, dim=0)
        ximea_labels = torch.tensor(ximea_labels, dtype=torch.int64)
        ximea_boxes = masks_to_boxes(ximea_masks)
    
    # Create the target for the ximea image
    x_target = {
        'boxes': ximea_boxes,
        'masks': ximea_masks,
        'labels': ximea_labels,
    }

    # Plot the original and predicted masks
    plot([(l_img, l_target_new), (x_img, x_target)], col_title=["Firefly Predicted Target", "Ximea FS Projected Target"])

    # Run the SAM2 predictor on the ximea image
    ximea_masks = sam_predictor.predict(x_img, x_target["boxes"], crop=False)
    x_target_new = x_target.copy()
    x_target_new['boxes'] = masks_to_boxes(ximea_masks)
    x_target_new['masks'] = ximea_masks
    plot([(x_img, x_target), (x_img, x_target_new)], col_title=["Ximea FS Projected Target", "Ximea SAM Predicted Target"])

    # # Save the annotations to a file
    # save_annotations_coco(
    #     os.path.join(DATA_DIR, "ximea", "annotations.json"),
    #     view_idx,
    #     new_ximea_target
    # )







if __name__ == '__main__':
    demo()
