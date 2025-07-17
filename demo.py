import os
import numpy as np
import torch

from utils.camera import Camera
from dataset import MultiCamDataset, SetType
from utils.visual import plot
from utils.utils import masks_to_boxes
from predictors import SAM2Predictor, FoundationStereoPredictor
from components import sample_frame, unproject_masks_to_3d, radius_outlier_filter, \
    transform_frame, project_to_plane, uvs_to_masks


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rivendale_dataset")

def demo(idx=5):
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

    # Check if the left image is empty
    if len(l_target['boxes']) == 0:
        print(f"Frame {idx}: No boxes found in the left image.")
        return

    # Pass the image and target to the SAM2 predictor
    l_masks = sam_predictor.predict(l_img, l_target["boxes"])
    l_target_new = l_target.copy()
    l_target_new['masks'] = l_masks
    plot([(l_img, l_target), (l_img, l_target_new)], col_title=["Original Target", "SAM Predicted Target"])
    
    # Get the depth map from the FoundationStereo predictor
    K = firefly_left.camera_matrix
    b = np.abs(firefly_left.transforms["firefly_right"][1][0])
    depth_map = fs_predictor.predict_depth(l_img, r_img, focal_length=K[0,0], baseline=b, scale=0.8, vis=True)

    # Get the masked list of points in 3D using the depth map
    pts3d = unproject_masks_to_3d(l_masks, depth_map, K)

    # Filter the points using the radius outlier filter
    pts3d_filtered = radius_outlier_filter(pts3d, radius=0.01, min_neighbors=100)

    # Transform the points to the ximea frame
    R, t = firefly_left.transforms["ximea"]
    x_pts3d = transform_frame(pts3d_filtered, R, t)

    # Project into the ximea image
    K2 = ximea.camera_matrix
    uv2s = project_to_plane(x_pts3d, K2)

    # Create the masks for the ximea image
    H, W = x_img[0].shape
    x_masks = uvs_to_masks(uv2s, H, W)
    
    # Create the target for the ximea image and replace the masks and boxes
    x_target = l_target_new.copy()
    x_target['boxes'] = masks_to_boxes(x_masks)
    x_target['masks'] = x_masks

    # Create a mask of valid masks based on the ratio of points in the mask
    theta = 0.5
    valid = [bool((torch.sum(x_masks[i]) / uv2s[i].shape[0]) > theta) for i in range(len(x_masks))]
    for k in x_target.keys():
        x_target[k] = x_target[k][valid]

    # Check if the target is empty
    if len(x_target['masks']) == 0:
        print(f"Frame {idx}: No valid masks found after filtering.")
        return

    # Plot the original and predicted masks
    plot([(l_img, l_target_new), (x_img, x_target)], col_title=["Firefly Predicted Target", "Ximea FS Projected Target"])

    # Run the SAM2 predictor on the ximea image
    ximea_masks = sam_predictor.predict(x_img, x_target["boxes"], crop=False)
    x_target_new = x_target.copy()
    x_target_new['boxes'] = masks_to_boxes(ximea_masks)
    x_target_new['masks'] = ximea_masks
    plot([(x_img, x_target), (x_img, x_target_new)], col_title=["Ximea FS Projected Target", "Ximea SAM Predicted Target"])

    # Distort the annotations back to the original image
    x_target_new = ximea.undistort_rectify_target(x_target_new, inverse=True)

    # Save the annotations to a file
    dataset.save_annos_coco(image_id=idx+1, target=x_target_new, cam=ximea)

if __name__ == '__main__':
    demo()


