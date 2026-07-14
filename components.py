import torch
import numpy as np
import open3d as o3d
from typing import List


def sample_frame(dataset, idx, cams):
    left = dataset[idx][cams[0].name]
    right= dataset[idx][cams[1].name]
    xim  = dataset[idx][cams[2].name]
    return left, right, xim

def unproject_masks_to_3d(masks: torch.Tensor, depth_map: np.ndarray,
    K: np.ndarray) -> List[np.ndarray]:
    """
    Given a list of binary masks (N, H, W) and a depth map (H, W),
    unproject each mask’s pixels into a point‐cloud in the camera frame.

    Args:
        masks:      torch.uint8 tensor of shape (N, H, W); nonzero→foreground.
        depth_map:  numpy array H×W of metric depths (same frame as masks).
        K:          3×3 camera intrinsics matrix.

    Returns:
        pts3d_list: list of length N; each is an (Mi×3) numpy array of 3D points.
    """
    K_inv = np.linalg.inv(K)
    pts3d_list = []

    for m in masks:
        m_np = m.cpu().numpy().astype(bool)
        vs, us = np.nonzero(m_np)
        zs = depth_map[vs, us]
        # stack into (M,3) of [u, v, z]
        uvz = np.stack([us, vs, zs], axis=1)
        # form homogeneous pixel coords [u, v, 1]
        homog = np.column_stack([uvz[:,0], uvz[:,1], np.ones_like(zs)])
        # backproject rays (M,3)
        rays = (K_inv @ homog.T).T
        # scale by depth to get (x,y,z)
        pts3d = rays * zs[:, None]
        pts3d_list.append(pts3d)

    return pts3d_list

def radius_outlier_filter(pts3d, radius=0.01, min_neighbors=100):
    """
    Filters a list of 3D point sets using a radius outlier filter.
    """
    # Create a list of filtered point sets
    filtered_pts3d = []
    for pts in pts3d:
        # Create a PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        # Apply radius outlier removal
        _, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        
        # Add the filtered points to the list
        filtered_pts3d.append(pts[ind])
        
    return filtered_pts3d

def filter_points_by_distance(pts3d_per_mask, distance_threshold):
    """
    Filters points in each mask based on a distance threshold.
    The distance is measured along the Z-axis in the camera coordinate system.
    """
    filtered_pts3d = []
    for pts3d in pts3d_per_mask:
        # The Z coordinate (depth) is the 3rd column
        depth = pts3d[:, 2]
        # Create a mask for points within the distance threshold
        mask = depth <= distance_threshold
        # Apply the mask
        filtered_pts3d.append(pts3d[mask])
    return filtered_pts3d

def transform_frame(pts3d_list: List[np.ndarray], R: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
    """
    Transform a list of point clouds to a new frame using a rotation and translation.

    Args:
        pts3d_list:    list of (Mi×3) numpy arrays.
        R:             3×3 rotation matrix.
        t:             3×1 translation vector.

    Returns:
        transformed:   list of (Ni×3) numpy arrays of the transformed points.
    """
    transformed = []
    for pts in pts3d_list:
        pts_transformed = (R @ pts.T + t.reshape(3, 1)).T
        transformed.append(pts_transformed)
    return transformed

def project_to_plane(pts3d_list: List[np.ndarray], K: np.ndarray) -> List[np.ndarray]:
    """
    Project a list of point clouds into the image plane using the camera intrinsics.

    Args:
        pts3d_list:    list of (Mi×3) numpy arrays.
        K:             3×3 camera intrinsics matrix.

    Returns:
        projected:     list of (Ni×2) numpy arrays of the projected points.
    """
    projected = []
    for pts in pts3d_list:
        uv_h = (K @ pts.T).T
        uv = uv_h[:, :2] / uv_h[:, [2]]
        projected.append(uv)
    return projected

def uvs_to_masks(uvs: List[np.ndarray], H: int, W: int) -> torch.Tensor:
    """
    Convert a list of 2D points to binary masks.

    Args:
        uvs: list of (Ni×2) numpy arrays of 2D points.
        H:   height of the image.
        W:   width of the image.

    Returns:
        masks: torch.uint8 tensor of shape (N, H, W) with binary masks.
    """
    masks = []
    for uv in uvs:
        # Create a mask for each set of points
        mask = torch.zeros((H, W), dtype=torch.uint8)
        us = torch.tensor(uv[:, 0].astype(int), dtype=torch.long)
        vs = torch.tensor(uv[:, 1].astype(int), dtype=torch.long)

        # Ensure the points are within the image bounds
        us = torch.clamp(us, 0, W - 1)
        vs = torch.clamp(vs, 0, H - 1)

        # Fill the mask
        mask[vs, us] = True
        masks.append(mask)

    # Stack the masks into a single tensor
    masks = torch.stack(masks, dim=0) if masks else torch.empty((0, H, W), dtype=torch.uint8)
    return masks