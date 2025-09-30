import numpy as np

# MediaPipe Pose 的 33 个关键点名称
POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

start_id =200
view_num=1
# 载入 npy 文件
data = np.load("/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/outputs/sq2/4_keypoints.npy", allow_pickle=True).item()

for i in range(0, view_num):
    k = start_id + i
    frame_id = f"frame_{k}"
    if frame_id in data:
        poses = data[frame_id]
        print(f"\n=== {frame_id} 共检测到 {len(poses)} 个舞者 ===")
        for idx, pose in enumerate(poses):
            print(f"\n舞者 {idx+1}:")
            normalized = pose["normalized"]
            world = pose["world"]
            for j, name in enumerate(POSE_LANDMARKS):
                n_x, n_y, n_z = normalized[j]
                w_x, w_y, w_z = world[j]
                print(f"{j:02d} {name:20s} "
                      f"Norm(x={n_x:.3f}, y={n_y:.3f}, z={n_z:.3f}) | "
                      f"World(x={w_x:.3f}, y={w_y:.3f}, z={w_z:.3f})")

    else:
        print(f"{frame_id} 不存在于数据中")
