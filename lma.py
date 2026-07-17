import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from physics_analyze_2D import process_view_2d_motion_from_file
from physics_analyze_2D import slice_result

FPS = 30
DT = 1.0 / FPS
HEIGHT = 1000  # cm，用于将 normalized -> cm 再转为 m
METER_SCALE = HEIGHT / 100.0  # 将 normalized * METER_SCALE 得到近似米

JOINTS_ORDER = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

BONE_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("NOSE", "LEFT_SHOULDER"),
    ("NOSE", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
]

# ---------- 参数（可调） ----------
LOW_PCT = 10
HIGH_PCT = 90
LABEL_LOW = 0.33
LABEL_HIGH = 0.66

# 选择用于 effort 计算的关键点索引（手/手腕 + 躯干）
EFFORT_JOINTS = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]

# ---------- 工具函数 ----------
def _clip01(x):
    return max(0.0, min(1.0, float(x)))

def _robust_minmax_scale(value_arr):
    """
    使用 clip 内的 LOW_PCT/HIGH_PCT 百分位做缩放，返回 [0,1]
    value_arr: 1D numpy
    """
    if np.isnan(value_arr).all():
        return lambda v: 0.0
    lo = np.percentile(value_arr, LOW_PCT)
    hi = np.percentile(value_arr, HIGH_PCT)
    if hi <= lo:
        # 避免除0
        def f(v):
            return 0.0
    else:
        def f(v):
            return _clip01((v - lo) / (hi - lo))
    return f

def _label_from_score(score, low_thresh=LABEL_LOW, high_thresh=LABEL_HIGH, low_label="Low", mid_label="Neutral", high_label="High"):
    if score >= high_thresh:
        return high_label
    if score <= low_thresh:
        return low_label
    return mid_label

# ---------- LMA 子指标计算函数 ----------
def _compute_body_part_participation(result, joints_order):
    """
    计算每个关节/分组的路径长度占比 -> 用于 Body 主导度判定
    返回：part_scores dict, per_joint_path (J,)
    """
    kp = result["kp_m"]              # (T,J,2)
    if kp.shape[0] < 2:
        raise ValueError("帧数不足")
    # per-joint path length (sum of euclidean distances across frames)
    diffs = np.linalg.norm(np.diff(kp, axis=0), axis=-1)   # (T-1, J)
    per_joint_path = diffs.sum(axis=0)                     # (J,)
    total = per_joint_path.sum() + 1e-12
    # groups
    groups = {
        "arms": ["LEFT_WRIST","RIGHT_WRIST","LEFT_ELBOW","RIGHT_ELBOW","LEFT_SHOULDER","RIGHT_SHOULDER"],
        "legs": ["LEFT_ANKLE","RIGHT_ANKLE","LEFT_KNEE","RIGHT_KNEE","LEFT_HEEL","RIGHT_HEEL"],
        "torso": ["LEFT_HIP","RIGHT_HIP","NOSE","LEFT_SHOULDER","RIGHT_SHOULDER"]
    }
    name2idx = {n:i for i,n in enumerate(joints_order)}
    part_scores = {}
    for gname, jlist in groups.items():
        idxs = [name2idx[s] for s in jlist if s in name2idx]
        part_scores[gname] = per_joint_path[idxs].sum() / total
    return part_scores, per_joint_path

def _compute_straightness(result, joints_order, joint_selection=None):
    """
    轨迹直线性（directness）:
    对每个选定关节，straightness = D / L where D = net displacement, L = path length
    返回每个关节的 straightness （J维），以及整体取均值
    """
    kp = result["kp_m"]  # (T,J,2)
    T = kp.shape[0]
    J = kp.shape[1]
    straightness = np.zeros(J, dtype=float)
    for j in range(J):
        p = kp[:, j, :]         # (T,2)
        diffs = np.linalg.norm(np.diff(p, axis=0), axis=-1)   # (T-1,)
        L = diffs.sum()
        D = np.linalg.norm(p[-1] - p[0])
        straightness[j] = 1.0 if L == 0 else (D / L)
    if joint_selection is None:
        mean_straight = np.nanmean(straightness)
    else:
        name2idx = {n:i for i,n in enumerate(joints_order)}
        idxs = [name2idx[s] for s in joint_selection if s in name2idx]
        mean_straight = float(np.nanmean(straightness[idxs])) if len(idxs)>0 else float(np.nanmean(straightness))
    return straightness, mean_straight

def _compute_time_suddenness(result, joints_order, joint_selection):
    """
    Suddenness (Time): 基于速度/加速度突发性。使用速度的峰度（kurtosis-like）与加速度峰值比例
    返回：一个标量 score (0-1) 越大越突发（sudden）
    """
    # 选择关节上的 speed 时间序列
    speed = result.get("speed")   # (T-1, J)
    if speed is None or speed.size==0:
        return 0.0
    name2idx = {n:i for i,n in enumerate(joints_order)}
    idxs = [name2idx[s] for s in joint_selection if s in name2idx]
    if len(idxs) == 0:
        idxs = list(range(speed.shape[1]))
    # aggregate scalar time-series (mean across selected joints)
    s_ts = np.mean(speed[:, idxs], axis=1)   # (T-1,)
    # 使用峰度近似：kurtosis = E[(x-mu)^4]/sigma^4 - 3
    mu = s_ts.mean()
    sigma = s_ts.std() + 1e-12
    kurt = np.mean(((s_ts - mu)**4)) / (sigma**4) - 3.0
    # 另计算加速度的短时突发占比
    accel = result.get("accel_mag")  # (T-2, J)
    if accel is None or accel.size==0:
        accel_peak_ratio = 0.0
    else:
        a_ts = np.mean(accel[:, idxs], axis=1)
        thr = np.percentile(a_ts, 75)
        accel_peak_ratio = float((a_ts > thr).sum()) / max(1, a_ts.size)
    # 组合：kurt 标准化再 minmax
    base = np.array([kurt, accel_peak_ratio])
    # scale each to 0-1 by intra-clip robust scaling
    f_kurt = _robust_minmax_scale(np.array([kurt,]))
    f_ratio = _robust_minmax_scale(np.array([accel_peak_ratio,]))
    s_kurt = f_kurt(kurt)
    s_ratio = f_ratio(accel_peak_ratio)
    score = 0.6 * s_kurt + 0.4 * s_ratio
    return _clip01(score)

def _compute_weight_strength(result, joints_order, joint_selection):
    """
    Weight (Strong vs Light): 用动能 proxy (0.5 * v^2) 的均值/峰值
    返回 0-1，越大越 Strong
    """
    speed = result.get("speed")   # (T-1, J)
    if speed is None or speed.size==0:
        return 0.0
    name2idx = {n:i for i,n in enumerate(joints_order)}
    idxs = [name2idx[s] for s in joint_selection if s in name2idx]
    if len(idxs) == 0:
        idxs = list(range(speed.shape[1]))
    energy_ts = 0.5 * (speed[:, idxs]**2)   # (T-1, K)
    # 使用均值与峰值组合
    mean_energy = float(np.mean(energy_ts))
    peak_energy = float(np.percentile(np.mean(energy_ts, axis=1), 90))
    scaler = _robust_minmax_scale(np.array([mean_energy, peak_energy]))
    # scale mean_energy and peak_energy individually and combine
    s_mean = scaler(mean_energy)
    s_peak = scaler(peak_energy)
    score = 0.5 * s_mean + 0.5 * s_peak
    return _clip01(score)

def _compute_flow_boundness(result, joints_order, joint_selection):
    """
    Flow (Bound vs Free): 使用 jerk（加加速度）大小与速度段断裂频率估计
    返回 0-1，越大越 Bound（约束）
    """
    kp = result["kp_m"]
    # compute jerk on selected joints: jerk = diff(acc)/DT
    # acc available: (T-2, J, 2) stored in result["acc"]
    acc = result.get("acc")   # (T-2,J,2)
    if acc is None or acc.size==0:
        return 0.0
    name2idx = {n:i for i,n in enumerate(joints_order)}
    idxs = [name2idx[s] for s in joint_selection if s in name2idx]
    if len(idxs)==0:
        idxs = list(range(acc.shape[1]))
    # compute jerk magnitudes
    jerk = np.diff(acc, axis=0) / DT   # (T-3, J, 2)
    jerk_mag = np.linalg.norm(jerk, axis=-1)   # (T-3, J)
    j_ts = np.mean(jerk_mag[:, idxs], axis=1) if jerk_mag.size>0 else np.array([0.0])
    # measure: mean jerk and zero-crossing count of velocity derivative (proxy for stops-starts)
    mean_jerk = float(np.mean(j_ts)) if j_ts.size>0 else 0.0
    vel = result.get("vel")   # (T-1,J,2)
    if vel is None or vel.size==0:
        zcr = 0.0
    else:
        # compute speed derivative sign changes as proxy for boundness
        speed_ts = np.mean(np.linalg.norm(vel[:, idxs, :], axis=-1), axis=1)  # (T-1,)
        dsign = np.sign(np.diff(speed_ts))
        zcr = float(((dsign[:-1] * dsign[1:]) < 0).sum()) / max(1, dsign.size)
    # robust scaling
    scaler = _robust_minmax_scale(np.array([mean_jerk, zcr]))
    s_j = scaler(mean_jerk)
    s_z = scaler(zcr)
    score = 0.6 * s_j + 0.4 * s_z
    return _clip01(score)

def _compute_kinesphere_and_shape(result, joints_order):
    """
    Shape measures:
      - kinesphere_size: mean/max distance from pelvis (use average hips as center)
      - rising: centroid y linear trend (positive -> rising)
      - widening: shoulder distance trend
    返回字典包含数值与对应 0-1 score（越大越明显）
    """
    kp = result["kp_m"]   # (T,J,2)
    name2idx = {n:i for i,n in enumerate(joints_order)}
    # pelvis center as mean of left/right hip if present
    if "LEFT_HIP" in name2idx and "RIGHT_HIP" in name2idx:
        pidx = [name2idx["LEFT_HIP"], name2idx["RIGHT_HIP"]]
        pelvis = kp[:, pidx, :].mean(axis=1)   # (T,2)
    else:
        pelvis = kp[:, 0, :]   # fallback to nose
    dist = np.linalg.norm(kp - pelvis[:, None, :], axis=-1)  # (T,J)
    kinesphere_mean = float(np.mean(dist))
    kinesphere_max = float(np.max(dist))
    # rising: use centroid (mean y of torso joints)
    torso_idxs = [name2idx[s] for s in ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"] if s in name2idx]
    centroid_y = kp[:, torso_idxs, 1].mean(axis=1) if len(torso_idxs)>0 else kp[:, :, 1].mean(axis=1)
    # fit linear slope
    t = np.arange(len(centroid_y))
    if len(t) >= 2:
        slope = float(np.polyfit(t, centroid_y, 1)[0])   # meters per frame
        slope_per_sec = slope / DT
    else:
        slope_per_sec = 0.0
    # widening: shoulder distance over time slope
    if "LEFT_SHOULDER" in name2idx and "RIGHT_SHOULDER" in name2idx:
        ls = kp[:, name2idx["LEFT_SHOULDER"], :]
        rs = kp[:, name2idx["RIGHT_SHOULDER"], :]
        shoulder_dist = np.linalg.norm(ls - rs, axis=-1)
        if len(shoulder_dist) >= 2:
            s_slope = float(np.polyfit(t, shoulder_dist, 1)[0]) / DT
        else:
            s_slope = 0.0
    else:
        s_slope = 0.0
    # scale outputs to 0-1
    ks_scaler = _robust_minmax_scale(np.array([kinesphere_mean, kinesphere_max]))
    kinesphere_score = ks_scaler(kinesphere_mean)
    rising_score = _clip01(_robust_minmax_scale(np.array([slope_per_sec]))(slope_per_sec))
    widening_score = _clip01(_robust_minmax_scale(np.array([s_slope]))(s_slope))
    return {
        "kinesphere_mean_m": kinesphere_mean,
        "kinesphere_max_m": kinesphere_max,
        "kinesphere_score": kinesphere_score,
        "rising_slope_m_per_s": slope_per_sec,
        "rising_score": rising_score,
        "shoulder_slope_m_per_s": s_slope,
        "widening_score": widening_score
    }

def _compute_space_directionality(result, joints_order):
    """
    Space: 方向性（是否有明确朝向）：
      - 计算选定参考点（双手或躯干）从首帧到末帧的位移向量，统计占比
      - directionality_score: dominant direction energy fraction (0-1)
    """
    kp = result["kp_m"]
    name2idx = {n:i for i,n in enumerate(joints_order)}
    refs = []
    for candidate in ["LEFT_WRIST","RIGHT_WRIST","NOSE","LEFT_SHOULDER"]:
        if candidate in name2idx:
            refs.append(name2idx[candidate])
    if len(refs) == 0:
        refs = [0]
    # compute displacements
    disp = kp[-1, refs, :] - kp[0, refs, :]   # (K,2)
    norms = np.linalg.norm(disp, axis=-1) + 1e-12
    # dominant direction is the one with largest norm
    total = norms.sum()
    dominant_frac = float(norms.max() / (total + 1e-12)) if total>1e-12 else 0.0
    # also compute global displacement direction concentration: project displacements to unit circle and compute length of mean vector
    unit = disp / norms[:, None]
    mean_vec = unit.mean(axis=0)
    concentration = float(np.linalg.norm(mean_vec))  # 0-1
    # score combine
    score = 0.6 * dominant_frac + 0.4 * concentration
    # scale robust
    score = _clip01(_robust_minmax_scale(np.array([score]))(score))
    return {
        "dominant_frac": dominant_frac,
        "concentration": concentration,
        "directionality_score": score
    }

# ---------- 主接口 ----------
def lma_analyze(result, joints_order=JOINTS_ORDER):
    """
    输入: result = process_view_2d_motion(...) 的返回值
    输出: lma dict 包含四个维度的 scores 与 labels，以及中间指标
    """
    # Body
    part_scores, per_joint_path = _compute_body_part_participation(result, joints_order)
    # determine dominant part
    dominant_part = max(part_scores.items(), key=lambda x: x[1])[0]

    # Effort.Space (directness)
    straightness_per_joint, mean_straight = _compute_straightness(result, joints_order, joint_selection=["LEFT_WRIST","RIGHT_WRIST"])
    direct_score = _clip01(_robust_minmax_scale(np.array([mean_straight]))(mean_straight))
    direct_label = "Direct" if direct_score >= LABEL_HIGH else ("Indirect" if direct_score <= LABEL_LOW else "Mixed")

    # Effort.Time (suddenness)
    time_score = _compute_time_suddenness(result, joints_order, EFFORT_JOINTS)
    time_label = "Sudden" if time_score >= LABEL_HIGH else ("Sustained" if time_score <= LABEL_LOW else "Mixed")

    # Effort.Weight (strong)
    weight_score = _compute_weight_strength(result, joints_order, EFFORT_JOINTS)
    weight_label = "Strong" if weight_score >= LABEL_HIGH else ("Light" if weight_score <= LABEL_LOW else "Mixed")

    # Effort.Flow (bound)
    flow_score = _compute_flow_boundness(result, joints_order, EFFORT_JOINTS)
    flow_label = "Bound" if flow_score >= LABEL_HIGH else ("Free" if flow_score <= LABEL_LOW else "Mixed")

    # Shape
    shape_info = _compute_kinesphere_and_shape(result, joints_order)
    # simple labels:
    shape_labels = []
    shape_labels.append("Expansive" if shape_info["kinesphere_score"] >= LABEL_HIGH else ("Contained" if shape_info["kinesphere_score"] <= LABEL_LOW else "Moderate"))
    shape_labels.append("Rising" if shape_info["rising_score"] >= LABEL_HIGH else ("Sinking" if shape_info["rising_score"] <= LABEL_LOW else "Stable"))
    shape_labels.append("Widening" if shape_info["widening_score"] >= LABEL_HIGH else ("Narrowing" if shape_info["widening_score"] <= LABEL_LOW else "Stable"))

    # Space
    space_info = _compute_space_directionality(result, joints_order)
    space_label = "Directional" if space_info["directionality_score"] >= LABEL_HIGH else ("Non-directional" if space_info["directionality_score"] <= LABEL_LOW else "Mixed")

    lma = {
        "Body": {
            "part_scores": part_scores,
            "dominant_part": dominant_part,
            "per_joint_path": per_joint_path
        },
        "Effort": {
            "Space": {"score": direct_score, "label": direct_label},
            "Time": {"score": time_score, "label": time_label},
            "Weight": {"score": weight_score, "label": weight_label},
            "Flow": {"score": flow_score, "label": flow_label}
        },
        "Shape": {
            "kinesphere": {"mean_m": shape_info["kinesphere_mean_m"], "max_m": shape_info["kinesphere_max_m"], "score": shape_info["kinesphere_score"], "label": shape_labels[0]},
            "rising": {"slope_m_s": shape_info["rising_slope_m_per_s"], "score": shape_info["rising_score"], "label": shape_labels[1]},
            "widening": {"shoulder_slope_m_s": shape_info["shoulder_slope_m_per_s"], "score": shape_info["widening_score"], "label": shape_labels[2]}
        },
        "Space": {
            **space_info,
            "label": space_label
        },
        "meta": {
            "score_thresholds": {"low": LABEL_LOW, "high": LABEL_HIGH},
            "percentile_scale": {"low_pct": LOW_PCT, "high_pct": HIGH_PCT}
        }
    }
    return lma

def generate_slice_index(frame_num,frames_per_slice):
    ii=0
    index_list = []
    while ii<frame_num:
        index_list.append((ii, min(ii+frames_per_slice-1, frame_num-1)))
        ii += frames_per_slice
    return index_list


def draw_lma_flowchart(segment_features_list, decimals=3, base_width=5, height=10):
    """
    将多个 segment 的 LMA 结果横向排成一行展示
    - 自动根据 segment 数量调整画布宽度
    - 每个 segment 一个方框
    """

    n = len(segment_features_list)
    
    # 自动设置窗口大小：每个 segment 分配 base_width 宽度
    fig_width = max(base_width * n, 8)
    fig, ax = plt.subplots(figsize=(fig_width, height))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_width = 0.85
    box_height = 1.0
    y0 = 0.075  # 每个 box 都在画布中央的水平线附近

    for i, feat in enumerate(segment_features_list):
        x0 = i + 0.075   # 每个 box 的 x 起点

        # 方框
        box = FancyBboxPatch(
            (x0, y0),
            box_width,
            box_height,
            boxstyle="round,pad=0.03",
            edgecolor="black",
            facecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(box)

        # Segment 标签（大号）
        ax.text(
            x0 + 0.05, y0 + box_height - 0.08,
            f"Segment {i+1}",
            fontsize=8,
            fontweight="bold",
            va="top"
        )

        # 内容区
        y_text = y0 + box_height - 0.18
        line_h = 0.02

        # ---------------- Body ----------------
        body = feat.get("Body", {})
        part_scores = body.get("part_scores", {})
        dominant_part = body.get("dominant_part", "unknown")

        ax.text(x0 + 0.05, y_text, "Body:", fontsize=8, fontweight="bold")
        y_text -= line_h

        for p, v in part_scores.items():
            ax.text(x0 + 0.07, y_text, f"{p}: {v:.{decimals}f}", fontsize=8)
            y_text -= line_h

        ax.text(x0 + 0.07, y_text, f"Dominant: {dominant_part}",
                fontsize=8, fontweight="bold")
        y_text -= line_h * 1.2

        # ---------------- Effort ----------------
        effort = feat.get("Effort", {})
        ax.text(x0 + 0.05, y_text, "Effort:", fontsize=8, fontweight="bold")
        y_text -= line_h

        for sub, info in effort.items():
            score = info.get("score", 0)
            label = info.get("label", "")
            ax.text(x0 + 0.07, y_text, f"{sub}: {score:.{decimals}f}", fontsize=8)
            y_text -= line_h * 0.8
            ax.text(x0 + 0.10, y_text, f"Label: {label}",
                    fontsize=8, fontweight="bold", color="darkred")
            y_text -= line_h * 1.1

        # ---------------- Shape ----------------
        shape = feat.get("Shape", {})
        ax.text(x0 + 0.05, y_text, "Shape:", fontsize=8, fontweight="bold")
        y_text -= line_h

        for sub, info in shape.items():
            score = info.get("score", 0)
            label = info.get("label", "")
            ax.text(x0 + 0.07, y_text, f"{sub}: {score:.{decimals}f}", fontsize=8)
            y_text -= line_h * 0.8
            ax.text(x0 + 0.10, y_text, f"Label: {label}",
                    fontsize=8, fontweight="bold", color="darkred")
            y_text -= line_h * 1.1

        # ---------------- Space ----------------
        space = feat.get("Space", {})
        ax.text(x0 + 0.05, y_text, "Space:", fontsize=8, fontweight="bold")
        y_text -= line_h

        for key in ["dominant_frac", "concentration", "directionality_score"]:
            if key in space:
                ax.text(x0 + 0.07, y_text,
                        f"{key}: {space[key]:.{decimals}f}", fontsize=8)
                y_text -= line_h

        ax.text(x0 + 0.07, y_text, f"Label: {space.get('label','')}",
                fontsize=8, fontweight="bold", color="darkred")

    plt.tight_layout()
    plt.show()


# ---------- 使用示例 ----------
if __name__ == "__main__":
    data_file = r"D:\Desktop\projects\Dance-Movements-Analyzer\mediapipe\datasets\251029whole_npy_final\technique\tech-daotizijinguan-perfect-1\merged_keypoints_ABC.npy"
    
    all_result = process_view_2d_motion_from_file(
        merged_npy_path=data_file,
        view="A_keypoints.npy"
    )
    frame_num=len(all_result["frames"])
    index_list=generate_slice_index(frame_num, 50)
    slice_lma_feature = []

    for cur_index in index_list:
        print(cur_index)
        cur_result = slice_result(all_result, cur_index)
        lma_feature = lma_analyze(cur_result)
        slice_lma_feature.append(lma_feature)
        print(lma_feature,'\n*************\n\n')

    draw_lma_flowchart(slice_lma_feature)