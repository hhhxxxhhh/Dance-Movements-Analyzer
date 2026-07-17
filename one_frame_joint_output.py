import os
import numpy as np
import matplotlib.pyplot as plt

POSE_CONNECTIONS = [
    # 头部
    ("NOSE", "LEFT_EYE_INNER"),
    ("LEFT_EYE_INNER", "LEFT_EYE"),
    ("LEFT_EYE", "LEFT_EYE_OUTER"),
    ("LEFT_EYE_OUTER", "LEFT_EAR"),

    ("NOSE", "RIGHT_EYE_INNER"),
    ("RIGHT_EYE_INNER", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
    ("RIGHT_EYE_OUTER", "RIGHT_EAR"),

    ("MOUTH_LEFT", "MOUTH_RIGHT"),

    # 上半身
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("LEFT_WRIST", "LEFT_PINKY"),
    ("LEFT_WRIST", "LEFT_INDEX"),
    ("LEFT_WRIST", "LEFT_THUMB"),

    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
    ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_THUMB"),

    # 躯干
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),

    # 下半身
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),

    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
]


POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

# 名字 -> index
INDEX = {name: i for i, name in enumerate(POSE_LANDMARKS)}

# 转成 index 形式骨架连接
POSE_CONNECTIONS_IDX = [(INDEX[a], INDEX[b]) for a, b in POSE_CONNECTIONS]


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
# debug folder for testing
# folder_paths = [r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy\emotion\emotion-angry-1"]

for folder_path in folder_paths:
    output_path = os.path.join(folder_path, "merged_keypoints_ABC.npy")
    data = np.load(output_path, allow_pickle=True).item()   # data: {frame_idx: {...}, ...}

    save_dir = os.path.join(folder_path, "A_keypoints_frames")
    os.makedirs(save_dir, exist_ok=True)

    # 按帧号排序（keys 可能是 int 或 str）
    frame_keys = sorted(data.keys(), key=lambda x: int(x))
    frames_list = []

    for k in frame_keys:
        record = data[k]
        # 有时没有 A 视角的记录，需容错
        if "A_keypoints.npy" not in record or not record["A_keypoints.npy"]:
            # 用全 0 占位（或跳过，视需求而定）
            # 这里选择跳过该帧
            continue

        a_entry = record["A_keypoints.npy"][0]   # list[0] -> dict
        if "normalized" not in a_entry:
            continue

        arr = np.asarray(a_entry["normalized"])   # expected shape (J, 3)
        # 若数据是 (3, J) 或转置的情况，也要兼容判断：
        if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] == len(POSE_LANDMARKS):
            # 转置为 (J,3)
            arr = arr.T

        if arr.shape[1] < 2:
            raise ValueError(f"normalized data has unexpected shape {arr.shape} at frame {k}")

        frames_list.append(arr[:, :2])  # 取 x,y -> (J,2)

    if len(frames_list) == 0:
        raise RuntimeError(f"No valid A frames found in {output_path}")

    # 将 list -> ndarray (F, J, 2)
    norm_xy = np.stack(frames_list, axis=0)
    num_frames, num_joints, _ = norm_xy.shape
    print("Loaded A_keypoints assembled:", norm_xy.shape)

    # 绘图并保存每帧
    for idx in range(num_frames):
        pts = norm_xy[idx]  # (J,2)
        x = pts[:, 0]
        y = pts[:, 1]

        plt.figure(figsize=(4, 6))
        plt.scatter(x, y, s=20)

        # 绘骨架（索引形式）
        for i, j in POSE_CONNECTIONS_IDX:
            if i < num_joints and j < num_joints:
                plt.plot([x[i], x[j]], [y[i], y[j]])

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"{idx:04d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print("Saved to:", save_dir)
