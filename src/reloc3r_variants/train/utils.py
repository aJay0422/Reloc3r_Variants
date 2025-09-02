import torch
import numpy as np
import cv2



def translation_angular_loss(pred_pose, gt_pose, reduction="mean", eps=1e-6):
    """
    Calculate the angular loss between the predicted and ground truth translation vectors.
    The angular loss is defined as the angle between the two vectors.
    :param pred_pose: torch.Tensor of shape (batch_size, 4, 4)
    :param gt_pose: torch.Tensor of shape (batch_size, 4, 4)
    :return: torch.Tensor of shape (batch_size,)
    """
    pred_t = pred_pose[:, :3, 3]
    gt_t = gt_pose[:, :3, 3]
    pred_t_norm = torch.norm(pred_t, dim=1)
    pred_t_norm = torch.clamp(pred_t_norm, min=eps)
    gt_t_norm = torch.norm(gt_t, dim=1)
    cos_theta = torch.sum(pred_t * gt_t, dim=1) / (pred_t_norm * gt_t_norm + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    # print("Smoothing term 1e-6 added to cos_theta")
    theta = torch.acos(cos_theta)

    if reduction == "mean":
        return theta.mean()
    elif reduction == "sum":
        return theta.sum()
    elif reduction == "none":
        return theta
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def rotation_angular_loss(pred_pose, gt_pose, reduction="mean", eps=1e-6):
    """
    Calculate the angular loss between the predicted and ground truth rotation matrices.
    The angular loss is defined as the angle between the two rotation matrices:
    $$\mathcal{l}_{R} = arccos(\frac{1}{2} (tr(R_{pred}^T R_{gt}) - 1))$$

    :param pred_pose: torch.Tensor of shape (batch_size, 4, 4)
    :param gt_pose: torch.Tensor of shape (batch_size, 4, 4)
    :return: torch.Tensor of shape (batch_size,)
    """
    pred_R = pred_pose[:, :3, :3]
    gt_R = gt_pose[:, :3, :3]
    trace = torch.diagonal(torch.matmul(pred_R.transpose(1, 2), gt_R), dim1=1, dim2=2).sum(dim=1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    if torch.isnan(theta).any():
        raise ValueError("NaN values found in the angular loss calculation.")
    if reduction == "mean":
        return theta.mean()
    elif reduction == "sum":
        return theta.sum()
    elif reduction == "none":
        return theta
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    

def get_rot_err(rot_a, rot_b):
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def get_transl_ang_err(dir_a, dir_b):
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)
    return err


def error_auc(rError, tErrors, thresholds):
    """
    Args:
        Error (list): [N,]
        tErrors (list): [N,]
        thresholds (list)
    """
    error_matrix = np.concatenate((rError[:, None], tErrors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20, 30]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

    

def get_aucs(pred_relposes, gt_relposes, return_separate=False):
    rot_errs = []
    t_errs = []
    for i in range(len(pred_relposes)):
        pred_pose = pred_relposes[i].detach().cpu().numpy()
        gt_pose = gt_relposes[i].detach().cpu().numpy()
        rot_err = get_rot_err(pred_pose[:3, :3], gt_pose[:3, :3])
        t_err = get_transl_ang_err(pred_pose[:3, 3], gt_pose[:3, 3])
        rot_errs.append(rot_err)
        t_errs.append(t_err)
    rot_errs = np.array(rot_errs)
    t_errs = np.array(t_errs)
    rot_errs[np.isnan(rot_errs)] = 180
    t_errs[np.isnan(t_errs)] = 180

    aucs = error_auc(rot_errs, t_errs, thresholds=[5, 10, 20])
    if return_separate:
        auc_rot = error_auc(rot_errs, np.zeros_like(t_errs), thresholds=[5, 10, 20])
        auc_transl = error_auc(np.zeros_like(rot_errs), t_errs, thresholds=[5, 10, 20])
        return aucs, auc_rot, auc_transl
    else:
        return aucs