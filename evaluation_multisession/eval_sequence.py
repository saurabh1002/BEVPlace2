from scipy.spatial.distance import cdist
import os
import sys
import open3d as o3d
import numpy as np
import pickle
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


def evaluate_sequence_reg(
    model, dataset_ref, dataset_query, ref_map_indices, query_map_indices, cfg
):
    closures = {}

    # Databases of previously visited/'seen' places.
    ref_descriptors_file = os.path.join(
        dataset_ref.sequence_dir, "bevplace2_descriptors.pkl"
    )
    with open(ref_descriptors_file, "rb") as dbfile2:
        ref_descriptors = pickle.load(dbfile2)

    db_descriptors = np.asarray(ref_descriptors).reshape(-1, 8192)

    for query_idx in trange(0, len(dataset_query), unit=" frames", dynamic_ncols=True):

        bev_image = getBEV(dataset_query[query_idx].astype(np.float64))
        bev_tensor = torch.from_numpy(bev_image).unsqueeze(0).float().cuda()
        _, _, global_descriptor = model(bev_tensor)
        global_descriptor = global_descriptor.detach().cpu().numpy()

        # Find top-1 candidate.
        feat_dists = cdist(
            global_descriptor, db_descriptors, metric=cfg.eval_feature_distance
        ).reshape(-1)
        candidate_indices = np.where(feat_dists <= cfg.thresh_max)[0]
        map_refs = ref_map_indices[candidate_indices]
        distances = feat_dists[candidate_indices]

        if len(candidate_indices) > 0:
            closures[query_map_indices[query_idx]] = (map_refs, distances)

    return closures
