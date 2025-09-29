model_path='/home/wuyou/hxh/mediapipe/models/pose_landmarker_heavy.task'

import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    """可视化关键点和骨架"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def main(input_video_path, output_video_path):
    # STEP 1: 初始化 PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='/home/wuyou/hxh/mediapipe/models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        num_poses=3 )  # 视频里关键点就够了
    detector = vision.PoseLandmarker.create_from_options(options)

    # 打开视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频保存
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_idx += 1

    #     # 按横宽比例选择ROI
    #     h, w, _ = frame.shape
    #     roi_ratio = 0.7  # 例如，选择宽度的50%
    #     roi_width = int(w * roi_ratio)
    #     roi_height = h  # 保持高度不变
    #     x_start = (w - roi_width) // 2  # 居中裁剪
    #     x_end = x_start + roi_width
    #     roi_frame = frame[:, x_start:x_end]  # 提取ROI

    #     # BGR → RGB
    #     rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #     # 推理
    #     detection_result = detector.detect(mp_image)
    #     if detection_result.segmentation_masks:
    #         seg_mask = detection_result.segmentation_masks[0].numpy_view()  # [H,W] float32, 0~1
    #         # 阈值化
    #         mask = seg_mask > 0.1

    #         # 背景替换为白色
    #         bg_image = 255 * np.ones_like(roi_frame, dtype=np.uint8)
    #         roi_frame = np.where(mask[..., None], roi_frame, bg_image)

    #     # 将处理后的ROI写回原始帧（可选）
    #     frame[:, x_start:x_end] = roi_frame

    #     # 保存视频帧
    #     out.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 推理
        detection_result = detector.detect(mp_image)
        if detection_result.segmentation_masks:
            seg_mask = detection_result.segmentation_masks[0].numpy_view()  # [H,W] float32, 0~1
            # 阈值化
            mask = seg_mask > 0.1

            # 背景替换为白色
            bg_image = 255 * np.ones_like(frame, dtype=np.uint8)
            frame = np.where(mask[..., None], frame, bg_image)

        # 可视化
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # 写入视频
        out.write(annotated_frame_bgr)

        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx} 帧...")

    cap.release()
    out.release()
    print(f"处理完成，结果保存在: {output_video_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python pose_landmark_video.py input.mp4 output.mp4")
    else:
        main(sys.argv[1], sys.argv[2])

