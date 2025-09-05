from scipy.spatial.distance import cdist
import os
import sys
import open3d as o3d
import numpy as np
from tqdm.auto import trange
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


def getBEV(scan):
    scan = scan[np.where(np.abs(scan[:, 0]) < 40)[0], :]
    scan = scan[np.where(np.abs(scan[:, 1]) < 40)[0], :]
    scan = scan[np.where(np.abs(scan[:, 2]) < 40)[0], :]

    pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan))
    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.4)

    scan = np.asarray(pointcloud.points)

    x_min = -40
    y_min = -40
    x_max = 40
    y_max = 40

    x_min_ind = np.floor(x_min / 0.4).astype(int)
    x_max_ind = np.floor(x_max / 0.4).astype(int)
    y_min_ind = np.floor(y_min / 0.4).astype(int)
    y_max_ind = np.floor(y_max / 0.4).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    mat_global_image = np.zeros((y_num, x_num), dtype=np.uint8)

    for i in range(scan.shape[0]):
        x_ind = x_max_ind - np.floor(scan[i, 1] / 0.4).astype(int)
        y_ind = y_max_ind - np.floor(scan[i, 0] / 0.4).astype(int)
        if x_ind >= x_num or y_ind >= y_num:
            continue
        if mat_global_image[y_ind, x_ind] < 10:
            mat_global_image[y_ind, x_ind] += 1

    mat_global_image = mat_global_image * 10

    mat_global_image[np.where(mat_global_image > 255)] = 255
    mat_global_image = mat_global_image / np.max(mat_global_image) * 255

    mat_global_image = (mat_global_image.astype(np.float32)) / 256
    mat_global_image = mat_global_image[np.newaxis, :, :].repeat(3, 0)

    return mat_global_image


def evaluate_sequence_reg(model, dataset, map_indices, cfg):
    # Databases of previously visited/'seen' places.
    global_descriptors = np.zeros((len(dataset), 8192))
    closures = {}

    for query_idx in trange(0, len(dataset), unit=" frames", dynamic_ncols=True):
        bev_image = getBEV(dataset[query_idx].astype(np.float64))
        bev_tensor = torch.from_numpy(bev_image).unsqueeze(0).float().cuda()
        _, _, global_descriptor = model(bev_tensor)
        global_descriptor = global_descriptor.detach().cpu().numpy()
        global_descriptors[query_idx] = global_descriptor

        map_query = map_indices[query_idx]
        if (query_idx) < 100:
            continue

        feat_dists = cdist(
            global_descriptor,
            global_descriptors[: query_idx - 100 + 1],
            metric=cfg.eval_feature_distance,
        ).reshape(-1)
        candidate_indices = np.where(feat_dists <= cfg.thresh_max)[0]
        map_refs = map_indices[candidate_indices]
        distances = feat_dists[candidate_indices]

        keep_indices = np.where(map_query - map_refs > 3)[0]

        if len(keep_indices) > 0:
            closures[map_query] = (map_refs[keep_indices], distances[keep_indices])

    return global_descriptors, closures
