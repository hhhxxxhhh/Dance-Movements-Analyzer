import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from scipy.stats import kurtosis

FPS=30

def finite_diff(arr, fps):
    dt = 1.0 / fps
    return (arr[1:] - arr[:-1]) / dt

def magnitude(v):
    return np.linalg.norm(v, axis=-1)

def compute_basic_kinematics(poses, fps):
    # poses: (T, J, 3)
    T, J, _ = poses.shape
    vel = np.zeros((T-1, J, 3))
    acc = np.zeros((T-2, J, 3))
    for j in range(J):
        vel[:, j, :] = finite_diff(poses[:, j, :], fps)
        acc[:, j, :] = finite_diff(vel[:, j, :], fps)
    speed = magnitude(vel)        # (T-1, J)
    accel = magnitude(acc)       # (T-2, J)
    jerk = np.abs(finite_diff(acc, fps))  # (T-3, J,3) -> take norm
    jerk = np.linalg.norm(jerk, axis=-1)  # (T-3, J)
    return {
        'vel': vel, 'acc': acc, 'jerk': jerk,
        'speed': speed, 'accel': accel
    }

def body_part_participation(poses, fps, joint_groups):
    # joint_groups: dict name -> list of joint indices
    kin = compute_basic_kinematics(poses, fps)
    speed = kin['speed']  # (T-1, J)
    total_path = speed.sum(axis=0)  # per-joint total path (sum speed)
    part_scores = {}
    for name, idxs in joint_groups.items():
        part_scores[name] = total_path[idxs].sum() / (total_path.sum() + 1e-8)
    return part_scores, total_path

def path_straightness(poses, joint_idx):
    p = poses[:, joint_idx, :]
    diffs = np.linalg.norm(p[1:] - p[:-1], axis=1)
    L = diffs.sum()
    D = np.linalg.norm(p[-1] - p[0])
    if L == 0: return 1.0
    return D / L

def kinesphere_size(poses):
    # use pelvis as center if available (assume index 0 as pelvis)
    center = poses[:, 0, :]  # adapt index to your skeleton
    rel = poses - center[:, None, :]
    dist = np.linalg.norm(rel, axis=-1)  # (T, J)
    return {
        'mean': dist.mean(),
        'max': dist.max(),
        'std': dist.std()
    }

def spatial_direction_histogram(poses, fps, joint_idx=0, nbins=8):
    # compute displacement vectors over whole clip for a reference joint
    p = poses[:, joint_idx, :]
    disp = p[-1] - p[0]
    # project to XY plane or 3D sphere bins; here simple azimuth bins:
    az = np.arctan2(disp[1], disp[0])  # radians
    bin_idx = int(((az + np.pi) / (2*np.pi)) * nbins) % nbins
    hist = np.zeros(nbins); hist[bin_idx] = 1.0
    return hist

def effort_scores_from_kin(poses, fps, joint_idxs=None):
    # aggregate over selected joints (e.g., hands + torso)
    kin = compute_basic_kinematics(poses, fps)
    speed = kin['speed'][:, joint_idxs].mean(axis=1) if joint_idxs is not None else kin['speed'].mean(axis=1)
    accel = kin['accel'][:, joint_idxs].mean(axis=1) if joint_idxs is not None else kin['accel'].mean(axis=1)
    jerk = kin['jerk'][:, joint_idxs].mean(axis=1) if joint_idxs is not None else kin['jerk'].mean(axis=1)

    # Time: sudden vs sustained -> use kurtosis of speed / accel peaks
    time_score = min(max(kurtosis(speed, fisher=False) - 3, 0), 10)  # higher -> more sudden
    # Weight: energy proxy
    energy = 0.5 * (speed**2).mean()
    weight_score = energy
    # Flow: bound vs free -> use mean jerk normalized
    flow_score = jerk.mean()
    # Space (directness): using average straightness across chosen joints
    if joint_idxs is not None:
        straits = [path_straightness(poses, j) for j in joint_idxs]
    else:
        straits = [path_straightness(poses, j) for j in range(poses.shape[1])]
    space_score = np.mean(straits)

    # normalize scores to 0-1 by heuristic or per-dataset stats: here return raw
    return {
        'time_raw': time_score,
        'weight_raw': weight_score,
        'flow_raw': flow_score,
        'space_raw': space_score
    }

# 示例：对一个 clip 提取特征向量
def extract_features_for_clip(poses, fps, joint_groups, selected_joints_for_effort):
    part_scores, per_joint_path = body_part_participation(poses, fps, joint_groups)
    ks = kinesphere_size(poses)
    eff = effort_scores_from_kin(poses, fps, selected_joints_for_effort)
    directions = spatial_direction_histogram(poses, fps, joint_idx=selected_joints_for_effort[0], nbins=8)

    feat = {}
    feat.update({'kinesphere_mean': ks['mean'], 'kinesphere_max': ks['max']})
    feat.update({'body_'+k: v for k, v in part_scores.items()})
    feat.update({k:v for k,v in eff.items()})
    feat.update({'dir_hist_'+str(i): directions[i] for i in range(len(directions))})
    # add global stats
    feat['total_path'] = per_joint_path.sum()
    return feat
