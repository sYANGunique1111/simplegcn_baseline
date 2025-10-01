from __future__ import absolute_import, division

import torch
import numpy as np

__all__ = ["mpjpe", "p_mpjpe", "PCK_loss", "AUC_loss"]


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


# def compute_PCK(gts, preds, scales=1000, eval_joints=None, threshold=150):
#     PCK_THRESHOLD = threshold
#     sample_num = len(gts)
#     total = 0
#     true_positive = 0
#     if eval_joints is None:
#         eval_joints = list(range(gts.shape[1]))

#     for n in range(sample_num):
#         gt = gts[n]
#         pred = preds[n]
#         # scale = scales[n]
#         scale = 1000
#         per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
#         true_positive += (per_joint_error < PCK_THRESHOLD).sum()
#         total += per_joint_error.size

#     pck = float(true_positive / total) * 100
#     return pck


def PCK_loss(gts, preds, scale=1000, threshold=150) -> float :
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts)
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    joint_loss = torch.sqrt(torch.sum(torch.pow(gts-preds, 2), dim=-1))
    # batch_loss = torch.sqrt(torch.sum(joint_loss, dim=1))
    true_positive = (joint_loss <= threshold).sum()
    num_joints = gts.shape[0] * gts.shape[1] 
    pck = float(true_positive / num_joints)
    # sample_num = len(gts)
    # total = 0
    # true_positive = 0
    # if eval_joints is None:
    #     eval_joints = list(range(gts.shape[1]))

    # for n in range(sample_num):
    #     gt = gts[n]
    #     pred = preds[n]
    #     # scale = scales[n]
    #     scale = 1000
    #     per_joint_error = np.take(np.sqrt(np.sum(np.power(pred - gt, 2), 1)) * scale, eval_joints, axis=0)
    #     true_positive += (per_joint_error < PCK_THRESHOLD).sum()
    #     total += per_joint_error.size

    # pck = float(true_positive / total) * 100
    return pck

def AUC_loss(gts, preds, scales=1000):
    # This range of thresholds mimics 'mpii_compute_3d_pck.m', which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = np.linspace(0, 150, 31)
    pck_list = []
    for threshold in thresholds:
        pck_list.append(PCK_loss(gts, preds, scales, threshold))

    auc = np.mean(pck_list)

    return auc


if __name__ == '__main__':
    tensor_shape = (64, 16, 3)
    g = torch.randn(tensor_shape)
    p = torch.randn(tensor_shape) 
    pck_loss = PCK_loss(g, p)
    # auc_loss = compute_AUC(g, p)
    # print(pck_loss, auc_loss)