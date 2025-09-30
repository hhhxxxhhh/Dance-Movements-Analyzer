import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
import re

frame_range = 70  # 帧间隔阈值
maxima_bottom = -0.3  # 局部最大值阈值

# Mediapipe 33个关键点
POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

def find_first_valid_peak(npy_path, frame_range, maxima_threshold,
                          joint_name="NOSE", coord_type="normalized",
                          smooth_window=5, poly_order=3):
    """
    查找第一个满足条件的局部最大值帧号
    条件: 帧号 < frame_range 且 Y 值 > maxima_threshold
    :param npy_path: npy 文件路径
    :param frame_range: 帧号阈值
    :param maxima_threshold: 峰值下限
    :param joint_name: 关键点名称
    :param coord_type: 'world' 或 'normalized'
    :param smooth_window: 平滑窗口长度
    :param poly_order: 多项式拟合阶数
    :return: 第一个符合条件的帧号 (int)，若无返回 None
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if joint_name not in POSE_LANDMARKS:
        raise ValueError(f"{joint_name} 不在关键点列表中")

    joint_idx = POSE_LANDMARKS.index(joint_name)
    frames, y_coords = [], []

    # 遍历帧
    for frame_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        poses = data[frame_key]
        if len(poses) == 0:
            continue
        # 取第一个舞者
        joint = poses[0][coord_type][joint_idx]
        _, y, _ = joint
        frames.append(int(frame_key.split('_')[1]))
        y_coords.append(-y)  # Mediapipe y 轴取反，向上为正

    y_coords = np.array(y_coords)

    # 平滑滤波
    if len(y_coords) >= smooth_window:
        y_smooth = savgol_filter(y_coords, smooth_window, poly_order)
    else:
        y_smooth = y_coords

    # 寻找局部最大值
    peaks, _ = find_peaks(y_smooth, distance=10)
    peak_frames = np.array(frames)[peaks]
    peak_values = y_smooth[peaks]

    # 筛选符合条件的峰值
    for f, val in zip(peak_frames, peak_values):
        if f < frame_range and val > maxima_threshold:
            return f
    return None

def y_peak_visualizer(npy_path, joint_name="NOSE", coord_type="normalized",
                       smooth_window=5, poly_order=3):
    """
    绘制某关节的Y坐标随帧数变化曲线，并加入Savitzky-Golay平滑滤波与峰值标注
    :param npy_path: npy 文件路径
    :param joint_name: 关键点名称
    :param coord_type: 'world' 或 'normalized'
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if joint_name not in POSE_LANDMARKS:
        print(f"错误: {joint_name} 不在关键点列表中")
        return
    
    joint_idx = POSE_LANDMARKS.index(joint_name)
    frames, y_coords = [], []

    # 遍历帧
    for frame_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        poses = data[frame_key]
        if len(poses) == 0:
            continue
        # 取第一个舞者
        joint = poses[0][coord_type][joint_idx]
        _, y, _ = joint
        frames.append(int(frame_key.split('_')[1]))
        y_coords.append(-y)  # 取反保证向上为正

    y_coords = np.array(y_coords)

    # 平滑滤波
    if len(y_coords) >= smooth_window:  # 防止数据太短
        y_smooth = savgol_filter(y_coords, smooth_window, poly_order)
    else:
        y_smooth = y_coords

    # 寻找局部最大值
    peaks, _ = find_peaks(y_smooth, distance=10)
    peak_frames = np.array(frames)[peaks]
    peak_values = y_smooth[peaks]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(frames, y_coords, label="original", alpha=0.5)
    plt.plot(frames, y_smooth, label="smoothed", color="red", linewidth=2)
    plt.scatter(peak_frames, peak_values, color="green", marker="o", label="local maxima")

    for f, val in zip(peak_frames, peak_values):
        plt.text(f, val, str(f), fontsize=8, ha="center", va="bottom")

    plt.xlabel("frame")
    plt.ylabel("Y")
    plt.title(f"Y of {joint_name} VS frame id")
    plt.legend()
    plt.grid(True)
    plt.show()


def align_and_merge_npy(folder_path, frame_range, maxima_threshold,output_path,
                        joint_name="NOSE", coord_type="normalized",
                        smooth_window=5, poly_order=3,
                        ):
    """
    遍历文件夹中所有 .npy 文件，找到对齐帧并合并成一个大 npy
    :param folder_path: 包含多个 .npy 的文件夹
    :param frame_range: find_first_valid_peak 参数
    :param maxima_threshold: find_first_valid_peak 参数
    :param joint_name: 关键点名称
    :param coord_type: 'world' 或 'normalized'
    :param output_path: 输出 npy 文件路径
    """
    npy_files = [f for f in os.listdir(folder_path) if re.match(r"^\d+_keypoints\.npy$", f)]
    print(f"找到 {len(npy_files)} 个 .npy 文件")
    if not npy_files:
        raise ValueError("文件夹中没有找到 .npy 文件")

    all_data = {}  # {frame_id: {filename: joint_data}}
    
    for npy_file in npy_files:
        npy_path = os.path.join(folder_path, npy_file)
        print(f"处理 {npy_file} ...")

        # 找到对齐帧号
        first_peak = find_first_valid_peak(
            npy_path, frame_range, maxima_threshold,
            joint_name=joint_name, coord_type=coord_type,
            smooth_window=smooth_window, poly_order=poly_order
        )
        if first_peak is None:
            print(f"⚠️ {npy_file} 未找到有效峰值，跳过")
            continue

        # 加载数据
        data = np.load(npy_path, allow_pickle=True).item()

        for frame_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
            old_frame = int(frame_key.split('_')[1])
            new_frame = old_frame - first_peak  # 对齐后的帧号

            if new_frame < 0:  # 丢弃对齐前的帧
                continue

            if new_frame not in all_data:
                all_data[new_frame] = {}
            
            # 存储时用文件名标识不同角度
            all_data[new_frame][npy_file] = data[frame_key]

    # 保存合并后的 npy
    np.save(output_path, all_data)
    print(f"✅ 已保存对齐后的数据到 {output_path}")

def makesure_threshold(folder_path, joint_name="NOSE", coord_type="normalized",):
    npy_files = [f for f in os.listdir(folder_path) if re.match(r"^\d+_keypoints\.npy$", f)]
    for npy_file in npy_files:
        y_peak_visualizer(npy_path=os.path.join(folder_path, npy_file),
                          joint_name=joint_name, coord_type=coord_type)


if __name__ == "__main__":
    npy_path = "/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/outputs/sq2/1_keypoints.npy"
    folder_path = "/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/outputs/sq2"
    output_path = "/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/outputs/sq2/merged_keypoints.npy"
    # makesure_threshold(folder_path=folder_path)
    align_and_merge_npy( folder_path=folder_path, frame_range=frame_range, maxima_threshold=maxima_bottom, output_path=output_path )
