import numpy as np
import matplotlib.pyplot as plt

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

# === 新增：易读骨段定义 ===
SEGMENTS = {
    "left_thigh": ("LEFT_HIP", "LEFT_KNEE"),
    "right_thigh": ("RIGHT_HIP", "RIGHT_KNEE"),

    "left_shank": ("LEFT_KNEE", "LEFT_ANKLE"),
    "right_shank": ("RIGHT_KNEE", "RIGHT_ANKLE"),

    "left_upper_arm": ("LEFT_SHOULDER", "LEFT_ELBOW"),
    "right_upper_arm": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),

    "left_forearm": ("LEFT_ELBOW", "LEFT_WRIST"),
    "right_forearm": ("RIGHT_ELBOW", "RIGHT_WRIST"),

    "body": ("LEFT_HIP", "LEFT_SHOULDER")
}


def load_2d_keypoints(merged_npy_path, view="A_keypoints.npy"):
    """
    从 merged npy 读取每帧的 normalized 2D (x,y)，但不做单位转换。
    返回 frames(list[int]) 与 keypoints_2d(list or None)，每项为 (J,2) numpy 或 None。
    """
    data = np.load(merged_npy_path, allow_pickle=True).item()
    frames = sorted(data.keys())
    keypoints_2d = []

    for f in frames:
        if view not in data[f]:
            keypoints_2d.append(None)
            continue

        entry = data[f][view]
        if not entry or not isinstance(entry, list) or "normalized" not in entry[0]:
            # 兼容不同存储形式
            try:
                arr3d = entry[0]
                arr2d = arr3d[:, :2]
            except Exception:
                keypoints_2d.append(None)
                continue
        else:
            arr3d = entry[0]["normalized"]   # shape=(33,3)
            arr2d = arr3d[:, :2]             # 只取前两个维度作为 2D
        keypoints_2d.append(arr2d.astype(float))

    return frames, keypoints_2d


def interpolate_missing_frames(frames, keypoints):
    """
    对 list of (J,2) 或 None 做逐关节逐维线性插值补齐。
    返回 numpy array (T, J, 2)。
    """
    kp_list = keypoints.copy()
    T = len(kp_list)
    # 找到第一个非 None 以确定 J
    first_valid = None
    for x in kp_list:
        if x is not None:
            first_valid = x
            break
    if first_valid is None:
        raise ValueError("所有帧均无关键点数据")

    J = first_valid.shape[0]
    # 初始化数组并标记有效位
    arr = np.full((T, J, 2), np.nan, dtype=float)
    valid = np.zeros((T, J), dtype=bool)
    for t in range(T):
        if kp_list[t] is None:
            continue
        arr[t] = kp_list[t]
        valid[t, :] = ~np.isnan(kp_list[t]).any(axis=1)  # 整关节行有效标记

    # 对每个关节、每个坐标维度插值
    idx = np.arange(T)
    for j in range(J):
        for c in range(2):
            seq = arr[:, j, c]
            good = ~np.isnan(seq)
            if good.sum() == 0:
                # 若整条序列都缺失，填 0（或使用该关节在第一有效帧的值）
                arr[:, j, c] = 0.0
                continue
            # 若首尾有缺失，np.interp 会做边界外延填充（使用最近有效值）
            arr[:, j, c] = np.interp(idx, idx[good], seq[good])
    return arr


def normalized_to_meters(kp_normalized):
    """
    将 normalized 坐标近似映射为米单位。
    这里使用 HEIGHT（人体高度 cm）作为尺度参考：
      physical_xy (meters) = normalized_xy * (HEIGHT/100)
    注意：这种近似假设图像归一化尺度与人体高度线性相关，若有相机内参请按真实尺度做变换。
    """
    return kp_normalized * METER_SCALE


def compute_velocity_acceleration(kp_m):
    """
    输入 kp_m: (T, J, 2)，单位为米
    输出：
      vel: (T-1, J, 2) 单位 m/s  ( = diff / DT )
      acc: (T-2, J, 2) 单位 m/s^2 ( = diff(vel) / DT )
      speed: (T-1, J) m/s
      accel_mag: (T-2, J) m/s^2
    """
    if kp_m.shape[0] < 2:
        return np.zeros((0, kp_m.shape[1], 2)), np.zeros((0, kp_m.shape[1], 2)), np.zeros((0, kp_m.shape[1])), np.zeros((0, kp_m.shape[1]))
    vel = (kp_m[1:] - kp_m[:-1]) / DT
    if vel.shape[0] < 2:
        acc = np.zeros((0, kp_m.shape[1], 2))
    else:
        acc = (vel[1:] - vel[:-1]) / DT
    speed = np.linalg.norm(vel, axis=-1)
    accel_mag = np.linalg.norm(acc, axis=-1)
    return vel, acc, speed, accel_mag


def compute_bone_vectors(kp_m, JOINTS_ORDER, BONE_PAIRS):
    """
    输入 kp_m (T,J,2)，返回每个骨对的向量（米）。
    输出 dict: (j1,j2) -> ndarray (T,2)
    """
    name_to_idx = {name: i for i, name in enumerate(JOINTS_ORDER)}
    T = kp_m.shape[0]
    bone_vecs = {}
    for j1, j2 in BONE_PAIRS:
        i1, i2 = name_to_idx[j1], name_to_idx[j2]
        bone_vecs[(j1, j2)] = kp_m[:, i2, :] - kp_m[:, i1, :]
    return bone_vecs

def compute_segment_vectors(kp_m, JOINTS_ORDER, SEGMENTS):
    """
    输入 kp_m (T,J,2)，返回各骨段的向量（米）
    输出 dict: name -> (T,2)
    """
    name_to_idx = {name: i for i, name in enumerate(JOINTS_ORDER)}
    seg_vecs = {}
    for seg_name, (j1, j2) in SEGMENTS.items():
        i1, i2 = name_to_idx[j1], name_to_idx[j2]
        seg_vecs[seg_name] = kp_m[:, i2, :] - kp_m[:, i1, :]
    return seg_vecs

def compute_segment_angle(vec1, vec2):
    """
    输入：
        vec1, vec2 : (T,2) 两个骨段的向量（米）
    输出：
        angles : (T,) 两骨段夹角（弧度 0~pi）
    """
    # dot = |a||b|cosθ
    dot = np.sum(vec1 * vec2, axis=-1)
    norm1 = np.linalg.norm(vec1, axis=-1)
    norm2 = np.linalg.norm(vec2, axis=-1)
    cos = dot / (norm1 * norm2 + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def compute_bone_angles_and_velocity(bone_vecs):
    """
    angles: (T,) 弧度，按 arctan2(y,x)
    ang_vel: (T-1,) rad/s (差分 / DT)
    返回两个 dict: angles[(j1,j2)] = (T,), ang_vel[(j1,j2)] = (T-1,)
    """
    angles = {}
    ang_vel = {}
    for pair, vec in bone_vecs.items():
        ang = np.arctan2(vec[:, 1], vec[:, 0])         # [-pi, pi]
        ang = np.unwrap(ang)                           # 避免跳跳变
        angles[pair] = ang
        if ang.shape[0] >= 2:
            ang_vel[pair] = np.diff(ang) / DT         # rad/s
        else:
            ang_vel[pair] = np.zeros((0,))
    return angles, ang_vel


def process_view_2d_motion_from_file(merged_npy_path, view="A_keypoints.npy"):
    frames, kp = load_2d_keypoints(merged_npy_path, view)
    kp_filled = interpolate_missing_frames(frames, kp)   # (T,J,2) in normalized units

    # 转为物理米单位
    kp_m = normalized_to_meters(kp_filled)  # (T,J,2) in meters

    # 速度/加速度/标量速度/加速度大小
    vel, acc, speed, accel_mag = compute_velocity_acceleration(kp_m)

    # 骨骼向量、角度与角速度（角速度已归一到 rad/s）
    bone_vecs = compute_bone_vectors(kp_m, JOINTS_ORDER, BONE_PAIRS)
    angles, ang_vel = compute_bone_angles_and_velocity(bone_vecs)

    seg_vecs = compute_segment_vectors(kp_m, JOINTS_ORDER, SEGMENTS)


    return {
        "frames": frames,            # list[int]
        "kp_norm": kp_filled,        # (T,J,2) 原始 normalized（插值后）
        "kp_m": kp_m,                # (T,J,2) 单位：米
        "vel": vel,                  # (T-1,J,2) 速度 单位：m/s
        "acc": acc,                  # (T-2,J,2) 加速度 单位：m/s^2
        "speed": speed,              # (T-1,J) 标量速度 单位：m/s
        "accel_mag": accel_mag,      # (T-2,J) 加速度大小 单位：m/s^2
        "bone_vecs": bone_vecs,      # {(j1,j2): (T,2)} 骨骼向量 单位：米
        "seg_vecs": seg_vecs,        # {name: (T,2)} 骨骼向量（易读名称查询） 单位：米
        "angles": angles,            # {(j1,j2): (T,)} 单位：rad
        "ang_vel": ang_vel           # {(j1,j2): (T-1,)} 关节夹角相对角速度 单位：rad/s
    }

def plot_motion_parameter_time_curve(time_axis, angle_series, title="Joint Angle", ylabel="Angle (deg)"):
    """
    绘制某关节角度随时间变化的曲线
    
    参数:
        time_axis: (T,) 时间轴（秒）或帧号
        angle_series: (T,) 角度数组，单位为弧度或角度
        title: 图标题
        ylabel: y轴名称
    """
    # 如果输入是弧度，自动转换为角度
    if np.max(angle_series) <= np.pi + 1e-3:
        angle_series_deg = angle_series * 180 / np.pi
    else:
        angle_series_deg = angle_series

    plt.figure(figsize=(10,4))
    plt.plot(time_axis, angle_series_deg)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def slice_result(result, frame_range):
    """
    对 result 的所有物理量按帧范围进行切片。
    frame_range: (start, end) 例如 (0, 50) 代表包含 0~50 帧。
    """
    start, end = frame_range
    end = min(end, len(result["frames"]) - 1)
    
    # === 基础帧切片 ===
    frames = result["frames"][start:end+1]

    # === 位置信息 ===
    kp_norm = result["kp_norm"][start:end+1]
    kp_m = result["kp_m"][start:end+1]

    # === 动态信息 ===
    # vel, speed: T-1  → 对应 [start, end-1]
    vel = result["vel"][start:end]
    speed = result["speed"][start:end]

    # acc, accel_mag: T-2 → 对应 [start, end-2]
    acc = result["acc"][start:end-1]
    accel_mag = result["accel_mag"][start:end-1]

    # === 骨骼类计算 ===
    bone_vecs = {}
    for k, v in result["bone_vecs"].items():   # v: (T,2)
        bone_vecs[k] = v[start:end+1]

    angles = {}
    for k, v in result["angles"].items():      # v: (T,)
        angles[k] = v[start:end+1]

    ang_vel = {}
    for k, v in result["ang_vel"].items():     # v: (T-1,)
        ang_vel[k] = v[start:end]

    # === 返回结构 ===
    return {
        "frames": frames,
        "kp_norm": kp_norm,
        "kp_m": kp_m,
        "vel": vel,
        "acc": acc,
        "speed": speed,
        "accel_mag": accel_mag,
        "bone_vecs": bone_vecs,
        "angles": angles,
        "ang_vel": ang_vel,
    }



if __name__ == "__main__":
    data_file = r"D:\Desktop\projects\Dance-Movements-Analyzer\mediapipe\datasets\251029whole_npy_final\technique\tech-daotizijinguan-normal-1\merged_keypoints_ABC.npy"
    result = process_view_2d_motion_from_file(
        merged_npy_path=data_file,
        view="A_keypoints.npy"
    )

    # 示例：左手腕速度（m/s）
    lwrist_idx = JOINTS_ORDER.index("LEFT_WRIST")
    left_wrist_vel = result["vel"][:, lwrist_idx]   # shape (T-1, 2)
    print("left_wrist_vel (m/s) head:", left_wrist_vel[:5])

    # 示例：左前臂角速度（rad/s）
    left_arm = ("LEFT_ELBOW", "LEFT_WRIST")
    print("left_arm ang_vel (rad/s) head:", result["ang_vel"][left_arm][:5])

    # 易读骨段命名，查询骨段方向向量
    left_thigh = result["seg_vecs"]["left_thigh"]     # (T,2)
    left_shank = result["seg_vecs"]["left_shank"]     # (T,2)
    right_thigh = result["seg_vecs"]["right_thigh"]   # (T,2)
    right_shank = result["seg_vecs"]["right_shank"]   # (T,2)
    body_ang_vec = result["seg_vecs"]["body"]   # (T,2)
    # 读取BONE_PAIRS中一段骨骼的角速度
    left_thigh_ang_vel = result["ang_vel"][("LEFT_HIP", "LEFT_KNEE")]

    left_knee_angle = compute_segment_angle(left_thigh, left_shank)  # (T,)
    right_knee_angle = compute_segment_angle(right_thigh, right_shank)  # (T,)
    hip_hip_angle = compute_segment_angle(left_thigh, right_thigh)
    body_thigh_angle = compute_segment_angle(body_ang_vec, left_thigh)

    time_axis = np.arange(len(left_knee_angle))*DT  
    time_axis_vel = np.arange(len(left_thigh_ang_vel))*DT

    # 绘图示例
    # 夹角与时间曲线
    plot_motion_parameter_time_curve(
        time_axis,
        body_thigh_angle,
        title="Body-Thigh Angle Over Time",
        ylabel="Angle (deg)"
    )
    # 关节角速度与时间曲线（time_axis_vel长度比time_axis短1）
    plot_motion_parameter_time_curve(
        time_axis_vel,
        left_thigh_ang_vel,
        title="Left Thigh Angular Velocity Over Time",
        ylabel="Angular Velocity (rad/s)"
    )


    # plot_motion_parameter_time_curve(
    #     time_axis,
    #     left_knee_angle,
    #     title="Left Knee Angle Over Time",
    #     ylabel="Angle (deg)"
    # )
    # plot_motion_parameter_time_curve(
    #     time_axis,
    #     right_knee_angle,
    #     title="Right Knee Angle Over Time",
    #     ylabel="Angle (deg)"
    # )
    # plot_motion_parameter_time_curve(
    #     time_axis,
    #     hip_hip_angle,
    #     title="Hip-Hip Angle Over Time",
    #     ylabel="Angle (deg)"
    # )
