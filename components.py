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

def radius_outlier_filter(pts3d_list: List[np.ndarray],
    radius: float, min_neighbors: int) -> List[np.ndarray]:
    """
    Apply Open3D radius outlier removal to each point cloud.

    Args:
        pts3d_list:    list of (Mi×3) numpy arrays.
        radius:        radius within which to search for neighbors (in same units).
        min_neighbors: minimum number of points within `radius` to keep a point.

    Returns:
        filtered:      list of (Ni×3) numpy arrays of the inlier points.
    """
    filtered = []
    for pts in pts3d_list:
        if pts.shape[0] == 0:
            filtered.append(pts)
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd_inliers, _ = pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )
        filtered.append(np.asarray(pcd_inliers.points))
    return filtered

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

def project_to_image(pts3d_list: List[np.ndarray], K: np.ndarray) -> List[np.ndarray]:
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