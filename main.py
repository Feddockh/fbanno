import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from utils.camera import Camera
from dataset import ReprojectionDataset
from utils.utils import masks_to_boxes
from predictors import SAM2Predictor, FoundationStereoPredictor
from components import sample_frame, unproject_masks_to_3d, radius_outlier_filter, \
    transform_frame, project_to_plane, uvs_to_masks


REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def process_frame(idx, cams, dataset, sam_predictor, fs_predictor, output_dir):
    # Sample an image from the dataset
    (l_img, l_target, _), (r_img, _, _), (x_img, x_target, x_img_path) = sample_frame(dataset, idx, cams)

    # Check if the left image is empty
    if len(l_target['boxes']) == 0:
        print(f"Frame {idx}: No boxes found in the left image.")
        return
    
    # Pass the image and target to the SAM2 predictor
    l_masks = sam_predictor.predict(l_img, l_target["boxes"])
    l_target_new = l_target.copy()
    l_target_new['masks'] = l_masks

    # Get the depth map from the FoundationStereo predictor
    K = cams[0].camera_matrix
    b = np.abs(cams[0].transforms[cams[1].name][1][0])
    depth_map = fs_predictor.predict_depth(l_img, r_img, focal_length=K[0,0], baseline=b, scale=0.8, vis=False)

    # Get the masked list of points in 3D using the depth map
    pts3d = unproject_masks_to_3d(l_masks, depth_map, K)

    # Filter the points using the radius outlier filter
    pts3d_filtered = radius_outlier_filter(pts3d, radius=0.01, min_neighbors=100)

    # Transform the points to the ximea frame
    R, t = cams[0].transforms[cams[2].name]
    x_pts3d = transform_frame(pts3d_filtered, R, t)

    # Project into the ximea image
    K2 = cams[2].camera_matrix
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
    
    # Run the SAM2 predictor on the ximea image
    ximea_masks = sam_predictor.predict(x_img, x_target["boxes"], crop=False)
    x_target_new = x_target.copy()
    x_target_new['boxes'] = masks_to_boxes(ximea_masks)
    x_target_new['masks'] = ximea_masks

    # Distort the annotations back to the original image
    x_target_new = cams[2].undistort_rectify_target(x_target_new, inverse=True)

    # Save the annotations to a file
    img_name = os.path.splitext(os.path.basename(x_img_path))[0]
    print(f"Saving annotations for {img_name}.txt")
    dataset.save_annos(img_name, target=x_target_new, cam=cams[2], output_dir=output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproject RGB annotations onto the NIR (ximea) camera.")
    parser.add_argument("--rgb-dir", required=True, help="Directory of RGB (firefly_left) images")
    parser.add_argument("--rgb-labels-dir", required=True, help="Directory of RGB YOLO label files (source annotations)")
    parser.add_argument("--right-dir", required=True, help="Directory of stereo-right (firefly_right) images")
    parser.add_argument("--nir-dir", required=True, help="Directory of NIR (ximea) images")
    parser.add_argument("--output-dir", default=os.path.join(REPO_DIR, "outputs", "reprojected_labels"),
                        help="Directory to write the reprojected NIR YOLO labels to")
    parser.add_argument("--calib-dir", default=None,
                        help="Directory of camera calibration YAMLs (default: repo's calibration_files/)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N frames")
    args = parser.parse_args()

    # Set up cameras once
    firefly_left  = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea         = Camera("ximea")
    cams = [firefly_left, firefly_right, ximea]
    for c in cams:
        c.load_params(calib_dir=args.calib_dir)
        c.compute_maps()

    # Initialize predictors once
    sam_predictor = SAM2Predictor("facebook/sam2-hiera-tiny")
    fs_predictor  = FoundationStereoPredictor(
        checkpoint_path=os.path.join(REPO_DIR, "FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"),
        config_path=os.path.join(REPO_DIR, "FoundationStereo/pretrained_models/23-51-11/cfg.yaml"),
    )

    # Create dataset
    dataset = ReprojectionDataset(args.rgb_dir, args.rgb_labels_dir, args.right_dir, args.nir_dir,
                                   cams, undistort_rectify=True)

    # Iterate with tqdm
    n_frames = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for idx in tqdm(range(n_frames), desc="Processing frames"):
        process_frame(idx, cams, dataset, sam_predictor, fs_predictor, args.output_dir)