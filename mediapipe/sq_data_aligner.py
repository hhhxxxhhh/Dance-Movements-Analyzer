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

def find_first_valid_peak(npy_path, front_n=30, threshold=0.05,
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

    if len(y_smooth) < front_n:
        baseline = np.mean(y_smooth)
    else:
        baseline = np.mean(y_smooth[:front_n])

    target_value = baseline + threshold

    # 寻找局部最大值
    peaks, _ = find_peaks(y_smooth, distance=10)
    peak_frames = np.array(frames)[peaks]
    peak_values = y_smooth[peaks]

    # 筛选符合条件的峰值
    for f, val in zip(peak_frames, peak_values):
        if val > target_value:
            print(npy_path, f)
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
    plt.title(f"Y of {joint_name} in {npy_path} \n VS frame id")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_merged_joint_y(merged_npy_path,
                             joint_name="NOSE", coord_type="normalized",
                             smooth_window=5, poly_order=3,
                             show_peaks=True):
    """
    对 align_and_merge_npy 输出的 merged.npy 文件可视化。
    一次性画出 A / B / C 三个视角的关节 Y 随帧变化曲线。

    :param merged_npy_path: 合并后的 npy 文件路径
    :param joint_name: 选择的关节
    :param coord_type: normalized 或 world
    :param smooth_window: SG 滤波窗口
    :param poly_order: SG 多项式阶数
    """
    data = np.load(merged_npy_path, allow_pickle=True).item()

    if joint_name not in POSE_LANDMARKS:
        print(f"{joint_name} 不在关键点列表内")
        return

    joint_idx = POSE_LANDMARKS.index(joint_name)

    views = ["A_keypoints.npy", "B_keypoints.npy", "C_keypoints.npy"]

    # 准备存储 A/B/C 的时序数据
    curves = {
        view: {"frames": [], "y": []}
        for view in views
    }

    # 遍历对齐后的帧
    for new_frame in sorted(data.keys()):
        for view in views:
            if view not in data[new_frame]:
                continue
            poses = data[new_frame][view]
            if len(poses) == 0:
                continue

            joint = poses[0][coord_type][joint_idx]
            _, y, _ = joint
            y = -y  # 保持向上为正

            curves[view]["frames"].append(new_frame)
            curves[view]["y"].append(y)

    # 画图
    plt.figure(figsize=(10, 12))
    for i, view in enumerate(views):
        frames = curves[view]["frames"]
        y_vals = np.array(curves[view]["y"])

        if len(frames) == 0:
            continue

        # 平滑
        if len(y_vals) >= smooth_window:
            y_smooth = savgol_filter(y_vals, smooth_window, poly_order)
        else:
            y_smooth = y_vals

        # 寻找峰值
        if show_peaks:
            peaks, _ = find_peaks(y_smooth, distance=10)
            peak_frames = np.array(frames)[peaks]
            peak_values = y_smooth[peaks]

        # 子图
        plt.subplot(3,1, i+1)
        plt.plot(frames, y_vals, alpha=0.5, label="original")
        plt.plot(frames, y_smooth, linewidth=2, label="smoothed")

        if show_peaks:
            plt.scatter(peak_frames, peak_values, color="green")
            for f, val in zip(peak_frames, peak_values):
                plt.text(f, val, str(f), fontsize=8, ha="center", va="bottom")

        plt.title(view.split("_")[0])
        plt.xlabel("frame")
        plt.ylabel("Y")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.suptitle(f"Y of {joint_name} in {merged_npy_path} \n Trajectories (A/B/C)")
    plt.tight_layout()
    plt.show()

def visualize_merged_joints_xy(
        merged_npy_path,
        joint_names=['NOSE','LEFT_WRIST','RIGHT_WRIST','LEFT_ANKLE','RIGHT_ANKLE'],
        coord_type="normalized",
        smooth_window=5,
        poly_order=3,
        show_peaks=False
    ):
    """
    一次绘制 6 张子图：
        1-3：A/B/C 的 X 曲线
        4-6：A/B/C 的 Y 曲线
    """
    data = np.load(merged_npy_path, allow_pickle=True).item()

    for jn in joint_names:
        if jn not in POSE_LANDMARKS:
            print(f"{jn} 不在关键点列表中")
            return

    views = ["A_keypoints.npy", "B_keypoints.npy", "C_keypoints.npy"]
    axes = ["X", "Y"]  # 绘图轴

    # curves[view][axis][joint] = { frames:[], vals:[] }
    curves = {
        view: {
            axis: {
                joint: {"frames": [], "vals": []}
                for joint in joint_names
            }
            for axis in axes
        }
        for view in views
    }

    # 解析数据
    for frame in sorted(data.keys()):
        frame_data = data[frame]

        for view in views:
            if view not in frame_data:
                continue

            poses = frame_data[view]
            if len(poses) == 0:
                continue

            for joint in joint_names:
                j_idx = POSE_LANDMARKS.index(joint)
                x, y, z = poses[0][coord_type][j_idx]

                curves[view]["X"][joint]["frames"].append(frame)
                curves[view]["X"][joint]["vals"].append(x)

                curves[view]["Y"][joint]["frames"].append(frame)
                curves[view]["Y"][joint]["vals"].append(-y)

    # 绘图：6 行，1 列
    plt.figure(figsize=(10, 9))

    subplot_id = 1
    for view in views:
        for axis in axes:

            plt.subplot(6, 1, subplot_id)
            subplot_id += 1

            all_frames = []

            for joint in joint_names:
                frames = curves[view][axis][joint]["frames"]
                values = np.array(curves[view][axis][joint]["vals"])

                if len(frames) == 0:
                    continue

                all_frames.extend(frames)

                if len(values) >= smooth_window:
                    smooth = savgol_filter(values, smooth_window, poly_order)
                else:
                    smooth = values

                plt.plot(frames, smooth, linewidth=1, label=joint)

                if show_peaks:
                    peaks, _ = find_peaks(smooth, distance=10)
                    pk_f = np.array(frames)[peaks]
                    pk_v = smooth[peaks]
                    plt.scatter(pk_f, pk_v)

            if len(all_frames) > 0:
                xmin, xmax = min(all_frames), max(all_frames)
                plt.xticks(np.arange(xmin, xmax + 1, 50))   # 每 50 帧标注
                plt.gca().set_xticks(np.arange(xmin, xmax + 1, 10), minor=True)
                plt.grid(which='minor', axis='x', linestyle='--', linewidth=0.5)

            plt.title(f"{view.split('_')[0]} - {axis}")
            plt.xlabel("frame")
            plt.ylabel(axis)
            plt.legend(fontsize=8)

    plt.suptitle(f"{merged_npy_path} \n Trajectories (A/B/C)")
    plt.tight_layout()
    plt.show()




def align_and_merge_npy(folder_path, front_n, threshold, output_path,
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
    npy_files = [f for f in os.listdir(folder_path) if re.match(r"^[ABC]+_keypoints\.npy$", f)]
    print(f"找到 {len(npy_files)} 个 .npy 文件")
    if not npy_files:
        raise ValueError("文件夹中没有找到 .npy 文件")

    all_data = {}  # {frame_id: {filename: joint_data}}
    
    for npy_file in npy_files:
        npy_path = os.path.join(folder_path, npy_file)
        print(f"处理 {npy_file} ...")

        # 找到对齐帧号
        first_peak = find_first_valid_peak(
            npy_path,
            front_n=front_n,
            threshold=threshold,
            joint_name=joint_name,
            coord_type=coord_type,
            smooth_window=smooth_window,
            poly_order=poly_order
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

            if npy_file == "A_keypoints.npy":
                # A 的帧率是其他视角的两倍，只保留偶数（或奇数，也可改）
                if new_frame % 2 != 0:
                    continue
                else:
                    new_frame = new_frame // 2  # 降采样

            if new_frame not in all_data:
                all_data[new_frame] = {}
            
            # 存储时用文件名标识不同角度
            all_data[new_frame][npy_file] = data[frame_key]

    # 保存合并后的 npy
    np.save(output_path, all_data)
    print(f"✅ 已保存对齐后的数据到 {output_path}")

def makesure_threshold(folder_path, joint_name="NOSE", coord_type="normalized",):
    npy_files = [f for f in os.listdir(folder_path) if re.match(r"^[ABC]+_keypoints\.npy$", f)]
    for npy_file in npy_files:
        y_peak_visualizer(npy_path=os.path.join(folder_path, npy_file),
                          joint_name=joint_name, coord_type=coord_type)


def delete_jumping_frame(
        src_root=r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy",
        dst_root=r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy_wo_jump",
        trim_frames=60
    ):
    """
    扫描 src_root 内所有 npy 文件，将其相对路径复制到 dst_root，
    并对内容进行截断（删除前 trim_frames 帧）。
    
    原路径： dataset/1/a/xxx.npy
    新路径： new_dataset/1/a/xxx.npy
    """

    for root, dirs, files in os.walk(src_root):
        for fname in files:
            if not fname.endswith("_ABC.npy"):
                continue

            src_path = os.path.join(root, fname)

            # 计算相对路径 例如 "1/a/xxx.npy"
            rel_path = os.path.relpath(src_path, src_root)

            # 目标路径
            dst_path = os.path.join(dst_root, rel_path)

            # 确保新目录存在
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # 读取数据
            data = np.load(src_path, allow_pickle=True).item()

            # 去除前 N 帧（key 为 int）
            trimmed = {
                (k - trim_frames): v
                for k, v in data.items()
                if k >= trim_frames
            }

            # 保存
            np.save(dst_path, trimmed)
            print(f"保存到 {dst_path}")

#去头
# if __name__ == "__main__":
#     delete_jumping_frame()

# 常用可视化绘图函数
if __name__ == "__main__":
    # npy_path = "D:\\Desktop\\projects\\DanceAnalyze\\datasets\\251029whole_npy\\emotion\\emotion-angry-1\\A_keypoints.npy"
    root = r"D:\Desktop\projects\Dance-Movements-Analyzer\mediapipe\datasets\251029whole_npy_final"
    folder_paths = []
    # 遍历 root 下所有子文件夹（仅一级）
    for sub1 in os.listdir(root):
        level1 = os.path.join(root, sub1)
        if not os.path.isdir(level1):
            continue

        for sub2 in os.listdir(level1):
            level2 = os.path.join(level1, sub2)
            if not os.path.isdir(level2):
                continue

            # 二级文件夹路径
            folder_paths.append(level2)

    #for debug
    folder_paths = [r"D:\Desktop\projects\Dance-Movements-Analyzer\mediapipe\datasets\251029whole_npy_final\emotion\emotion-angry-1"]

    for folder_path in folder_paths:
        output_path = os.path.join(folder_path, "merged_keypoints_ABC.npy")
        print(f"Processing folder: {folder_path}")
        # align_and_merge_npy(folder_path=folder_path, front_n=200, threshold=0.04, output_path=output_path)
        # makesure_threshold(folder_path=folder_path)
        # visualize_merged_joint_y(merged_npy_path=output_path,joint_name='NOSE')
        visualize_merged_joints_xy(merged_npy_path=output_path)

#去尾
# if __name__ == "__main__":

#     root = r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy_wo_jump"
#     dst_root = r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy_final"

#     folder_paths = []
#     # 遍历 root 下所有子文件夹（仅二级）
#     for sub1 in os.listdir(root):
#         level1 = os.path.join(root, sub1)
#         if not os.path.isdir(level1):
#             continue

#         for sub2 in os.listdir(level1):
#             level2 = os.path.join(level1, sub2)
#             if not os.path.isdir(level2):
#                 continue

#             folder_paths.append(level2)

#     #单独更新
#     folder_paths=[r'D:\Desktop\projects\Dance-Movements-Analyzer\mediapipe\datasets\251029whole_npy_wo_jump\technique\tech-daotizijinguan-perfect-1']

#     for folder_path in folder_paths:
#         src_path = os.path.join(folder_path, "merged_keypoints_ABC.npy")
#         if not os.path.exists(src_path):
#             print(f"{src_path} 不存在，跳过")
#             continue

#         print(f"\nProcessing folder: {folder_path}")
#         # 可视化
#         visualize_merged_joints_xy(merged_npy_path=src_path)

#         # 用户输入裁剪帧号
#         try:
#             trim_frame = int(input("请输入保留帧数，输入0表示不裁剪: \n"))
#         except ValueError:
#             print("输入无效，跳过裁剪")
#             trim_frame = 0

#         if trim_frame > 0:
#             # 计算相对路径
#             rel_path = os.path.relpath(src_path, root)
#             dst_path = os.path.join(dst_root, rel_path)
#             os.makedirs(os.path.dirname(dst_path), exist_ok=True)

#             # 加载数据
#             data = np.load(src_path, allow_pickle=True).item()

#             # 保留 trim_frame 之前的帧
#             trimmed = {k: v for k, v in data.items() if k <= trim_frame}

#             # 保存到新路径
#             np.save(dst_path, trimmed)
#             print(f"裁剪后数据已保存到 {dst_path}")
#         else:
#             print("未进行裁剪")
