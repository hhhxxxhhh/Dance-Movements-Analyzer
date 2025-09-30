# Dance-Movements-Analyzer
本项目基于 **MediaPipe PoseLandmarker**，实现舞蹈视频的人体关键点检测、骨架可视化及关键点数据保存（npy 格式）。  

---

## 目录结构

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

## 路径配置说明
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
### 从原始视频中进行数据提取
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