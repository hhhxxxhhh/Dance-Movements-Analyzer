import numpy as np
import matplotlib.pyplot as plt
import sys

# 33个关键点名称
POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

def plot_joint_trajectory_3d(npy_path, joint_name="RIGHT_WRIST", coord_type="world"):
    """
    绘制指定关节的三维运动轨迹
    :param npy_path: npy 文件路径
    :param joint_name: 关键点名称
    :param coord_type: 'world' 或 'normalized'
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if joint_name not in POSE_LANDMARKS:
        print(f"错误: {joint_name} 不在关键点列表中")
        return

    joint_idx = POSE_LANDMARKS.index(joint_name)

    xs, ys, zs = [], [], []

    for frame_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        poses = data[frame_key]
        if len(poses) == 0:
            continue
        # 这里取第一个舞者，如果有多舞者可以改
        joint = poses[0][coord_type][joint_idx]
        x, y, z = joint
        xs.append(x)
        ys.append(-y)
        zs.append(z)

    # 绘制轨迹
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='o', markersize=3, label=joint_name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{joint_name} trajectory ({coord_type} coordinates)")
    ax.legend()
    plt.show()

def plot_joint_trajectory_2d(npy_path, joint_name="RIGHT_WRIST", coord_type="world"):
    """
    绘制指定关节的二维运动轨迹
    :param npy_path: npy 文件路径
    :param joint_name: 关键点名称
    :param coord_type: 'world' 或 'normalized'
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if joint_name not in POSE_LANDMARKS:
        print(f"错误: {joint_name} 不在关键点列表中")
        return

    joint_idx = POSE_LANDMARKS.index(joint_name)

    xs, ys = [], []

    for frame_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        poses = data[frame_key]
        if len(poses) == 0:
            continue
        # 这里取第一个舞者，如果有多舞者可以改
        joint = poses[0][coord_type][joint_idx]
        x, y, _ = joint  # 忽略 z 坐标
        xs.append(x)
        ys.append(-y)

    # 绘制轨迹
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker='o', markersize=3, label=joint_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{joint_name} trajectory ({coord_type} coordinates)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python points_traj_viewer.py keypoints.npy [JOINT_NAME] [coord_type]")
        print("默认 JOINT_NAME=RIGHT_WRIST, coord_type=world")
        sys.exit(1)

    npy_file = sys.argv[1]
    joint_name = sys.argv[2] if len(sys.argv) >= 3 else "RIGHT_WRIST"
    coord_type = sys.argv[3] if len(sys.argv) >= 4 else "world"
    plot_joint_trajectory_2d(npy_file, joint_name, coord_type)
    # plot_joint_trajectory_3d(npy_file, joint_name, coord_type)