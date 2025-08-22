import os
import numpy as np
import torch
from tqdm import tqdm

from utils.camera import Camera
from dataset import YoloV5MultiCamDataset, SetType
from utils.utils import masks_to_boxes
from predictors import SAM2Predictor, FoundationStereoPredictor
from components import sample_frame, unproject_masks_to_3d, filter_points_by_distance, project_to_plane, uvs_to_masks
from utils.visual import plot
from torchvision.utils import save_image

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "rivendale_v5")

def process_frame(idx, cams, dataset, sam_predictor, fs_predictor, distance_threshold):
    # Sample an image from the dataset
    (l_img, l_target, l_img_path), (r_img, _, _), _ = sample_frame(dataset, idx, cams)

    # Check if the left image is empty
    if len(l_target['boxes']) == 0:
        print(f"Frame {idx}: No boxes found in the left image.")
        return
    
    # Pass the image and target to the SAM2 predictor
    l_masks = sam_predictor.predict(l_img, l_target["boxes"])

    # Get the depth map from the FoundationStereo predictor
    K = cams[0].camera_matrix
    b = np.abs(cams[0].transforms[cams[1].name][1][0])
    depth_map = fs_predictor.predict_depth(l_img, r_img, focal_length=K[0,0], baseline=b, scale=0.8, vis=False)

    # Create a mask for pixels within the distance threshold
    depth_mask = torch.from_numpy(depth_map <= distance_threshold)

    # Create a new image with only the pixels within the distance threshold
    l_img_filtered = l_img.clone()
    # l_img_filtered[:, ~depth_mask] = 0 # Zero out pixels beyond the threshold

    l_img_filtered = torch.where(
        depth_mask.unsqueeze(0),
        l_img_filtered,
        torch.tensor(BACKGROUND_COLOR, dtype=l_img_filtered.dtype).view(-1, 1, 1)
    )
    
    # Get the masked list of points in 3D using the depth map
    pts3d_per_mask = unproject_masks_to_3d(l_masks, depth_map, K)

    # Filter points beyond the distance threshold
    filtered_pts3d_per_mask = filter_points_by_distance(pts3d_per_mask, distance_threshold)

    # Project filtered points back to the image plane
    uvs_per_mask = project_to_plane(filtered_pts3d_per_mask, K)

    # Create new masks from the projected points
    H, W = l_img[0].shape
    new_masks = uvs_to_masks(uvs_per_mask, H, W)

    # Compare l_masks and new_masks and remove masks which have been occluded
    OCCLUSION_THRESHOLD = 50  # percentage
    new_masks_filtered = []
    for i, (l_mask, new_mask) in enumerate(zip(l_masks, new_masks)):
        if torch.sum(l_mask) > 0:
            occlusion = ((torch.sum(l_mask) - torch.sum(new_mask)) / torch.sum(l_mask)) * 100
        else:
            occlusion = 0
        if occlusion > OCCLUSION_THRESHOLD:
            print(f"Mask {i} has been occluded ({occlusion:.1f}% occluded). Removing mask.")
        else:
            new_masks_filtered.append(new_mask)
    
    if not new_masks_filtered:
        print(f"Frame {idx}: No masks left after occlusion filtering.")
        new_masks = torch.empty((0, H, W), dtype=torch.uint8)    
    else:
        new_masks = torch.stack(new_masks_filtered)

    # Create new target with filtered masks and new boxes
    new_target = l_target.copy()
    new_target['masks'] = new_masks
    new_target['boxes'] = masks_to_boxes(new_masks)

    # Filter out empty masks and corresponding annotations
    valid_indices = [i for i, mask in enumerate(new_masks) if torch.sum(mask) > 0]
    if not valid_indices:
        print(f"Frame {idx}: No valid masks found after filtering.")
        return
        
    for k in new_target.keys():
        new_target[k] = new_target[k][valid_indices]

    # Show the new images with thier filtered annotations
    # plot([(l_img, l_target), (l_img_filtered, new_target)], col_title=["Original Target", "Filtered Target"])

    # Save the new image
    img_name = os.path.splitext(os.path.basename(l_img_path))[0]
    print(f"Saving image for {img_name}.png")
    
    # Normalize image to [0, 1] if it's in [0, 255] range
    if l_img_filtered.max() > 1.0:
        l_img_filtered = l_img_filtered / 255.0
    
    save_image(l_img_filtered, os.path.join(OUTPUT_DIR, f"{img_name}.png"))

    # Save the new annotations
    img_name = os.path.splitext(os.path.basename(l_img_path))[0]
    print(f"Saving annotations for {img_name}.txt")
    dataset.save_annos(img_name, target=new_target, cam=firefly_left_filtered)

if __name__ == '__main__':
    DISTANCE_THRESHOLD = 0.6  # meters
    BACKGROUND_COLOR = (114.0, 114.0, 114.0) # [0.0, 255.0]
    SETTYPE = SetType.VAL

    # Set up cameras once
    firefly_left  = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea         = Camera("ximea")
    cams = [firefly_left, firefly_right, ximea]
    for c in cams:
        c.load_params()
        c.compute_maps()

    # Create the filtered left firefly image
    firefly_left_filtered = firefly_left.copy()
    firefly_left_filtered.name = f"{firefly_left_filtered.name}_filtered"

    OUTPUT_DIR = os.path.join(DATA_DIR, firefly_left_filtered.name, "images", SETTYPE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize predictors once
    sam_predictor = SAM2Predictor("facebook/sam2-hiera-tiny")
    fs_predictor  = FoundationStereoPredictor(
        checkpoint_path="FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
        config_path="FoundationStereo/pretrained_models/23-51-11/cfg.yaml",
    )

    # Create dataset
    dataset = YoloV5MultiCamDataset(DATA_DIR, cams, set_type=SETTYPE, undistort_rectify=True)

    # Iterate with tqdm
    for idx in tqdm(range(len(dataset)), desc="Processing frames"):
        # idx = 20
        process_frame(idx, cams, dataset, sam_predictor, fs_predictor, DISTANCE_THRESHOLD)
        # break
