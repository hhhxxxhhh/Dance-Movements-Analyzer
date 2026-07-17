# Dance-Movements-Analyzer
本项目基于 **MediaPipe PoseLandmarker**，实现舞蹈视频的人体关键点检测、骨架可视化及关键点数据保存（npy 格式）。  

---

## 目录结构

**文件结构有删改，部分目录变动，忽略下面文件结构，只需关注完成“掐头去尾、对齐”等预处理后的npy在mediapipe\datasets\251029whole_npy_final，动作描述文本文件在standard\动作打标(0709)**
工程文件目录为：
```
xxxxx/xxx/Dance-Movements-Analyzer/
├── mediapipe/
│ ├── models/
│ │ └── pose_landmarker_heavy.task # PoseLandmarker 模型文件
│ ├── datasets/ # 视频数据文件夹
│ │ └── sq2/1.mp4 # 示例视频
│ ├── outputs/ # 输出文件夹（视频和关键点 npy）
│ ├── test_data_saved.py # 主脚本
│ ├── test.py # (临时文件，可忽略)
│ ├── points_traj_viewer.py # 指定关节运动轨迹绘制工具
│ ├── points_video_cmp.py # 单帧检查工具
│ └── view_data.py # keypoints.npy预览工具
└── README.md
```



## 依赖环境

建议使用 Python 3.9+，并安装以下依赖：

```bash
pip install opencv-python mediapipe numpy tqdm
```
- opencv-python：视频读取和绘制

- mediapipe：人体关键点检测

- numpy：保存关键点数据

- tqdm：显示处理进度

## 路径配置说明（有变动，忽略）
在 test_data_saved.py 中，有几个重要的路径变量：

```
base_input_dir = "/xxxxxx/Dance-Movements-Analyzer/mediapipe/datasets"
base_output_dir = "/xxxxxx/Dance-Movements-Analyzer/mediapipe/outputs"
model_path = '/xxxxxx/Dance-Movements-Analyzer/mediapipe/models/pose_landmarker_heavy.task'
```
- base_input_dir：存放原始视频文件的根目录

- base_output_dir：处理后的视频和关键点数据保存目录

- model_path：PoseLandmarker 模型文件路径

请根据实际路径修改这三个变量。

## 运行方法

### 曲线绘图说明
1. 关节夹角
运行 physics_analyze_2D.py，函数process_view_2d_motion_from_file中进行预处理，返回值、查询键值和对应含义如下

'''
    # (T,J,2) 每个关节每个时刻的二维坐标
    # (T-1,J,2)是每个关节每个时刻的速度向量
    # (j1,j2)是BONE_PAIRS列表中的一种
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
'''

读取键值并绘制随时间变化的曲线图方法如下

'''
    result = process_view_2d_motion_from_file(
        merged_npy_path=data_file,
        view="A_keypoints.npy"
    )

    # 易读骨段命名，查询骨段方向向量
    left_thigh = result["seg_vecs"]["left_thigh"]     # (T,2)
    left_shank = result["seg_vecs"]["left_shank"]     # (T,2)

    # 读取BONE_PAIRS中一段骨骼的角速度
    left_thigh_ang_vel = result["ang_vel"][("LEFT_HIP", "LEFT_KNEE")]

    # 计算左膝关节两侧骨段夹角
    left_knee_angle = compute_segment_angle(left_thigh, left_shank)  # (T,)
    
    # 时间轴
    time_axis = np.arange(len(left_knee_angle))*DT  
    time_axis_vel = np.arange(len(left_thigh_ang_vel))*DT

    # 绘图示例
    # 夹角与时间曲线
    plot_motion_parameter_time_curve(
        time_axis,
        left_knee_angle,
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
'''

2. 关节路径
```
python mediapipe/points_traj_viewer.py {path to .npy file}/xxxx.npy <关节名称> normalized
```
例如绘制sq2/4.mp4中nose的轨迹：
```
python mediapipe/points_traj_viewer.py mediapipe/outputs/sq2/4_keypoints.npy NOSE normalized
```
在mediapipe/points_traj_viewer.py文件的最后，可以通过修改注释选择绘制X-Y二维投影轨迹plot_joint_trajectory_2d，或者绘制X-Y-Z三维空间轨迹plot_joint_trajectory_2d

3. 关节横向偏移（X）高度（Y）随时间变化曲线
参考mediapipe\sq_data_aligner.py中451行开始没有被注释的“常用可视化”过程，主要函数visualize_merged_joints_xy

### 从原始视频中进行数据提取【可忽略，此处npy数据已进行预处理，一般不再使用】
每一个全新的视频需要经过以下流程的处理，获取原始动作数据：
```
python mediapipe/test_data_saved.py <视频相对路径>
```
<视频相对路径>：相对于 datasets 文件夹的路径，例如：

```
python mediapipe/test_data_saved.py sq2/1.mp4
```
即对/xxxxxx/Dance-Movements-Analyzer/mediapipe/datasets/sq2/1.mp4这一视频进行处理，处理结果包括：
- 输出视频：outputs/sq2/1_text.mp4，包含可视化骨架和坐标信息
- 关键点数据：outputs/sq2/1_keypoints.npy，npy格式数据
  - 数据格式：
    ```
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
  - 关键点顺序：
    ```
    POSE_LANDMARKS = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX"
    ]
- **本项目多用归一化坐标** 
### npy数据预览方法

打开mediapipe/view_data.py,修改路径为实际.npy文件路径
```
data = np.load("{your path to project}/Dance-Movements-Analyzer/mediapipe/outputs/{.npy to be previewed}", allow_pickle=True).item()
```
然后命令行运行
```
python mediapipe/view_data.py
```
可以自定义预览范围
```
start_id = 0 #预览的起始帧
view_num = 20 #打印帧数
```
### 单个关节轨迹可视化
```
python mediapipe/points_traj_viewer.py {path to .npy file}/xxxx.npy <关节名称> normalized
```
例如绘制sq2/4.mp4中nose的轨迹：
```
python mediapipe/points_traj_viewer.py mediapipe/outputs/sq2/4_keypoints.npy NOSE normalized
```
在mediapipe/points_traj_viewer.py文件的最后，可以通过修改注释选择绘制X-Y二维投影轨迹plot_joint_trajectory_2d，或者绘制X-Y-Z三维空间轨迹plot_joint_trajectory_2d

### 单帧抽取-检查工具
```
python mediapipe/points_video_cmp.py {path to .npy file}/xxxx.npy <原始视频.mp4> <第x帧> <第x个舞者> normalized"
```
例如检查sq2/4.mp4中第200帧的图像和关节状态
```
python mediapipe/points_video_cmp.py  mediapipe/outputs/sq2/4_keypoints.npy mediapipe/datasets/sq2/4.mp4 200 0 normalized
```

### 视频数据对齐
1. 确定直立蹦跳帧阈值
   
   main中运行函数
   ```
   makesure_threshold(folder_path=folder_path)
   ```
   folder_path配置为一个sequence下的outputs，包含每个视角的.npy数据。
   
   此时会输出每个视角经过平滑处理的鼻子轨迹，观察直立跳的结束帧数和最高点纵坐标，按下面要求配置范围阈值

   ```
   frame_range = 70     # 小于每个视角正式动作开始的帧数
   maxima_bottom = -0.3 # 小于直立跳最高点纵坐标，大于其他被标记的局部最大值
   ```
2. 对齐并合并.npy数据
   
   ！！！两个函数中读取.npy的文件名格式需要重新调整，与你的.npy格式适配
   
   main中运行函数
   ```
   align_and_merge_npy( folder_path=folder_path, frame_range=frame_range, maxima_threshold=maxima_bottom, output_path=output_path )
   ```
   outputpath为合并npy路径，eg.xxx/xxxx/merged_keypoints.npy
   merged_keypoints.npy字典结构:
   ```
   # frame_id 为int
    {
        frame_id:{
            '4_keypoints.npy':
            [
                {
                    'normalized': array,
                    'world': array
                }
            ],
            '1_keypoints.npy':[...],
            '2_keypoints.npy':[...],
            '3_keypoints.npy':[...]
        },
        frame_id:{...},
        frame_id:{...}
    }
    ```


## 数据示例和输出demo
详见网盘链接链接: 
https://pan.baidu.com/s/1VCunGhMYEjwyNciFk83bFQ?pwd=dkcg 提取码: dkcg 

下载后按照“目录结构”所述，将文件放入对应目录。

## 注意事项
- 请确保视频路径正确，否则脚本会报错无法读取视频。
- 输出文件会覆盖同名文件，请注意备份原数据。