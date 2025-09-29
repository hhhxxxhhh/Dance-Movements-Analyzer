import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# Mediapipe Pose 33 个关键点名称
POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

def plot_frame_and_skeleton(npy_path, video_path, frame_id=100, person_idx=0, coord_type="world"):
    """
    分别显示指定帧的 3D 骨架 和 原视频帧
    :param npy_path: npy 文件路径
    :param video_path: 原视频路径
    :param frame_id: 帧号 (从0开始)
    :param person_idx: 舞者索引 (0表示第一个舞者)
    :param coord_type: 'world' 或 'normalized'
    """
    # 加载关键点数据
    data = np.load(npy_path, allow_pickle=True).item()
    frame_key = f"frame_{frame_id}"
    if frame_key not in data:
        print(f"帧 {frame_id} 不存在")
        return

    poses = data[frame_key]
    if person_idx >= len(poses):
        print(f"舞者 {person_idx} 不存在 (当前帧共有 {len(poses)} 个舞者)")
        return

    coords = poses[person_idx][coord_type]
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    # 打开视频并定位到指定帧
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"无法读取视频的第 {frame_id} 帧")
        return

    # 在视频帧上绘制关键点（归一化坐标 -> 像素）
    if "normalized" in poses[person_idx]:
        h, w, _ = frame.shape
        norm_coords = poses[person_idx]["normalized"]
        for i, (nx, ny, _) in enumerate(norm_coords):
            cx, cy = int(nx * w), int(ny * h)
            cv2.circle(frame,(500,0),10,(255,0,0),-1)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            if 'ANKLE' in POSE_LANDMARKS[i]:
                print('break')
            cv2.putText(frame, POSE_LANDMARKS[i], (cx+3, cy-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    # ------------------- 图 1：视频帧 -------------------
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Video Frame {frame_id}")
    plt.axis("off")

    # ------------------- 图 2：3D 骨架 -------------------
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='red', s=40)
    for i, name in enumerate(POSE_LANDMARKS):
        ax.text(xs[i], ys[i], zs[i], name, fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Skeleton - Frame {frame_id} - Person {person_idx} ({coord_type})")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python points_video_cmp.py keypoints.npy video.mp4 [frame_id] [person_idx] [coord_type]")
        print("默认 frame_id=100, person_idx=0, coord_type=world")
    else:
        npy_path = sys.argv[1]
        video_path = sys.argv[2]
        frame_id = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        person_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        coord_type = sys.argv[5] if len(sys.argv) > 5 else "world"
        plot_frame_and_skeleton(npy_path, video_path, frame_id, person_idx, coord_type)
