import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm  # 进度条
import os,sys

# 模型路径
model_path = '/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/models/pose_landmarker_heavy.task'

# 关节点名称
POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

# npy 文件保存格式
"""
npy 文件的数据结构说明：

xxx_keypoints.py 是一个字典，键为每一帧的标识，值为该帧检测到的舞者关节点信息列表。

结构示例：
{
    "frame_1": [   # 第一帧
        {   # 第一个舞者
            "normalized": np.ndarray,  # 归一化坐标 (Nx3)，N=33 个关键点
                                        # 每个关键点顺序对应 POSE_LANDMARKS 列表
                                        # 坐标范围：
                                        #   x: [0,1]，相对于图像宽度
                                        #   y: [0,1]，相对于图像高度
                                        #   z: 相对于相机平面的深度，通常为负数表示在相机前方
            "world": np.ndarray        # 世界坐标 (Nx3)，单位与摄像头坐标系一致
                                        # 原点位于相机坐标系或模型默认原点
                                        # x: 右侧为正，y: 上方为正，z: 前方为负
        },
        {   # 第二个舞者（如果存在）
            "normalized": ...,
            "world": ...
        },
        ...
    ],
    "frame_2": [...],
    ...
}

关键点顺序：
POSE_LANDMARKS = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

使用示例：
data = np.load("your_keypoints.npy", allow_pickle=True).item()
frame_10_first_pose_norm = data["frame_10"][0]["normalized"]
frame_10_first_pose_world = data["frame_10"][0]["world"]
"""

# 格式化文本
def format_pose_text(frame_idx, poses):
    lines = [f"=== frame_{frame_idx} {len(poses)} dancer(s) are detected  ==="]
    for pid, pose in enumerate(poses):
        lines.append(f"\nDancer {pid+1}:")
        for i, name in enumerate(POSE_LANDMARKS):
            nx, ny, nz = pose["normalized"][i]
            wx, wy, wz = pose["world"][i]
            line = f"{i:02d} {name:<18} Norm(x={nx:.3f}, y={ny:.3f}, z={nz:.3f}) | World(x={wx:.3f}, y={wy:.3f}, z={wz:.3f})"
            lines.append(line)
    return lines

# 渲染文字到图像
def render_text_panel(lines, width, height):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = 25, 22
    for i, line in enumerate(lines):
        y = y0 + i * dy
        if y > height - 10:
            break
        cv2.putText(panel, line, (10, y), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return panel

# 使用官方绘制方法绘制骨架
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

def main():
    if len(sys.argv) != 2:
        print("用法: python pose_landmark_video.py <视频相对路径，例如 sq2/1.mp4>")
        sys.exit(1)

    video_rel_path = sys.argv[1]  # 例如 sq2/1.mp4
    base_input_dir = "/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/datasets"
    base_output_dir = "/home/wuyou/hxh/Dance-Movements-Analyzer/mediapipe/outputs"

    input_video_path = os.path.join(base_input_dir, video_rel_path)
    # 输出视频路径：将 sq2/1.mp4 -> outputs/sq2/1_text.mp4
    output_video_path = os.path.join(base_output_dir, os.path.splitext(video_rel_path)[0] + "_text.mp4")
    # npy 路径：outputs/sq2/1_keypoints.npy
    output_npy_path = os.path.join(base_output_dir, os.path.splitext(video_rel_path)[0] + "_keypoints.npy")

    # 创建输出文件夹（如果不存在）
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height*2))

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    all_frames_keypoints = {}
    frame_idx = 0

    with tqdm(total=total_frames, desc="处理进度", unit="帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # 保存关键点
            frame_keypoints = []
            for pose_idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
                normalized = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
                world = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.pose_world_landmarks[pose_idx]])
                frame_keypoints.append({"normalized": normalized, "world": world})
            all_frames_keypoints[f"frame_{frame_idx}"] = frame_keypoints

            # 文字面板
            lines = format_pose_text(frame_idx, frame_keypoints)
            text_panel = render_text_panel(lines, width, height)

            # 拼接上下图像
            combined = np.vstack((annotated_frame_bgr, text_panel))
            out.write(combined)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    np.save(output_npy_path, all_frames_keypoints)
    print(f"视频保存至: {output_video_path}")
    print(f"关键点保存至: {output_npy_path}")

if __name__ == "__main__":
    main()
