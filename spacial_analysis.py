import numpy as np
import matplotlib.pyplot as plt
# 导入 3D 绘图模块
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter  # 引入平滑滤波器

# MediaPipe Pose 的 33 个关键点官方定义名称（仅作参考和说明）
POSE_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]


def calculate_3d_position_normalized(pt1, pt2, cam2_pos, f1_norm, f2_norm, cam_height=1.35, img_width=2560, img_height=1440):
    """
    修正了画面长宽比的 3D 重建算法
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x0, y0 = cam2_pos
    
    cx, cy = 0.5, 0.5
    
    # 计算长宽比 (Aspect Ratio)
    aspect_ratio = img_height / img_width
    
    C1 = np.array([0.0, 0.0, cam_height])
    C2 = np.array([-x0, y0, cam_height])
    
    # ⚠️ 注意这里：对 (cy - y) 乘以了 aspect_ratio
    V1 = np.array([
        x1 - cx, 
        f1_norm, 
        (cy - y1) * aspect_ratio
    ], dtype=np.float64)
    V1 = V1 / np.linalg.norm(V1)
    
    # ⚠️ 左侧相机同理
    V2 = np.array([
        f2_norm, 
        cx - x2, 
        (cy - y2) * aspect_ratio
    ], dtype=np.float64)
    V2 = V2 / np.linalg.norm(V2)
    
    # 求解最短距离连线中点
    W0 = C1 - C2
    a, b, c = np.dot(V1, V1), np.dot(V1, V2), np.dot(V2, V2)
    d, e = np.dot(W0, V1), np.dot(W0, V2)
    
    denom = a * c - b * b
    if denom < 1e-6:
        return np.array([0.0, 0.0, 0.0]), 999.0
        
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    P1 = C1 + t1 * V1
    P2 = C2 + t2 * V2
    
    estimated_pt = (P1 + P2) / 2.0
    error_distance = np.linalg.norm(P1 - P2)
    
    return estimated_pt, error_distance


def reconstruct_and_save_3d(
    input_npy_path, 
    output_raw_npy_path,       # 原始数据保存路径
    output_smooth_npy_path,    # 平滑后数据保存路径
    front_view_key="A_keypoints.npy", 
    left_view_key="B_keypoints.npy",
    f1_norm=1.0, 
    f2_norm=1.0, 
    cam2_pos=(2.0, 2.0),
    cam_height=1.35,
    smooth_window=11,          # 滤波窗口大小 (必须是奇数，越大越平滑)
    smooth_poly=3              # 多项式拟合阶数
):
    """
    读取 2D 对齐数据，进行三维重建。
    保存一份原始 3D 坐标，并利用 SG 滤波器进行时序平滑后，另存一份平滑版坐标。
    """
    print(f"正在加载二维坐标数据: {input_npy_path}")
    data_2d = np.load(input_npy_path, allow_pickle=True).item()
    
    data_3d_raw = {}
    total_error = 0.0
    valid_joints = 0
    
    # 用于收集时序数据以便进行平滑滤波
    # valid_frames 记录有数据的帧号，all_raw_coords 形状为 (N_frames, 33, 3)
    valid_frames = []
    all_raw_coords = []
    
    # ==========================================
    # 第一阶段：计算原始 3D 坐标并保存
    # ==========================================
    for frame in sorted(data_2d.keys()):
        frame_data = data_2d[frame]
        
        if front_view_key not in frame_data or left_view_key not in frame_data:
            continue
            
        front_poses = frame_data[front_view_key]
        left_poses = frame_data[left_view_key]
        
        if len(front_poses) == 0 or len(left_poses) == 0:
            continue
            
        front_kpts = front_poses[0]["normalized"]
        left_kpts = left_poses[0]["normalized"]
        
        world_3d_kpts = []
        
        for idx in range(33):
            pt_front = front_kpts[idx][:2] 
            pt_left  = left_kpts[idx][:2]
            
            pos_3d, err = calculate_3d_position_normalized(
                pt1=pt_front, pt2=pt_left, cam2_pos=cam2_pos,
                f1_norm=f1_norm, f2_norm=f2_norm, cam_height=cam_height
            )
            
            world_3d_kpts.append(pos_3d.tolist())
            total_error += err
            valid_joints += 1

        data_3d_raw[frame] = {
            "Reconstructed_3D": [{"world": world_3d_kpts}]
        }
        valid_frames.append(frame)
        all_raw_coords.append(world_3d_kpts)

    # 1. 导出原始结果
    np.save(output_raw_npy_path, data_3d_raw)
    
    avg_err = (total_error / valid_joints) * 100 if valid_joints > 0 else 0
    print(f"✅ 第一步：原始 3D 数据重建完成，共 {len(valid_frames)} 帧。")
    print(f"   保存至: {output_raw_npy_path}")
    print(f"   拟合平均误差: {avg_err:.2f} 厘米")

    # ==========================================
    # 第二阶段：对全序列 3D 坐标进行时序平滑处理
    # ==========================================
    if len(valid_frames) == 0:
        print("❌ 没有提取到有效帧数据，平滑处理中止。")
        return

    # 转换为 NumPy 数组，方便按轴平滑 (形状: [Frames, 33, 3])
    raw_coords_np = np.array(all_raw_coords)
    
    # 确保滤波窗口大小合法 (必须是奇数，且不能超过数据长度)
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window > len(valid_frames):
        smooth_window = len(valid_frames)
        if smooth_window % 2 == 0: smooth_window -= 1
        
    data_3d_smoothed = {}

    if smooth_window > smooth_poly:
        # 对 axis=0 (时间维度) 进行 Savitzky-Golay 滤波
        # 它会自动独立处理 33个关节 的 X, Y, Z 坐标
        smoothed_coords_np = savgol_filter(raw_coords_np, window_length=smooth_window, polyorder=smooth_poly, axis=0)
    else:
        print("⚠️ 数据帧数太少，无法进行平滑处理，保持原样。")
        smoothed_coords_np = raw_coords_np
        
    # 重组回原始的嵌套字典结构
    for i, frame in enumerate(valid_frames):
        data_3d_smoothed[frame] = {
            "Reconstructed_3D": [{"world": smoothed_coords_np[i].tolist()}]
        }

    # 2. 导出平滑后的结果
    np.save(output_smooth_npy_path, data_3d_smoothed)
    print(f"✅ 第二步：3D 轨迹平滑处理完成 (窗口大小: {smooth_window}, 多项式阶数: {smooth_poly})。")
    print(f"   保存至: {output_smooth_npy_path}")

def generate_laban_kinesphere(center, scale=0.9):
    """
    内部辅助函数：生成拉班空间的几何点位和平面数据
    """
    S = scale
    phi = (1 + np.sqrt(5)) / 2  # 黄金分割率 1.618
    
    a = S
    b = S / phi
    
    # 1. 人体中心点
    c_pt = np.array([[0, 0, 0]])
    # 2. 基础方向 (6个)
    cardinals = np.array([
        [S, 0, 0], [-S, 0, 0], [0, S, 0], [0, -S, 0], [0, 0, S], [0, 0, -S]
    ])
    # 3. 对角线方向 (8个)
    diagonals = np.array([
        [S, S, S], [S, S, -S], [S, -S, S], [S, -S, -S],
        [-S, S, S], [-S, S, -S], [-S, -S, S], [-S, -S, -S]
    ])
    # 4. 二十面体顶点 (12个)
    icosahedron_pts = np.array([
        [0, a, b], [0, a, -b], [0, -a, b], [0, -a, -b],
        [b, 0, a], [-b, 0, a], [b, 0, -a], [-b, 0, -a],
        [a, b, 0], [-a, b, 0], [a, -b, 0], [-a, -b, 0]
    ])
    
    # 平移到中心点
    c_pt = c_pt + center
    cardinals = cardinals + center
    diagonals = diagonals + center
    icosahedron_pts = icosahedron_pts + center
    
    # 计算二十面体连线
    target_dist = 2 * b
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(np.linalg.norm(icosahedron_pts[i] - icosahedron_pts[j]), target_dist, atol=1e-3):
                edges.append((i, j))
                
    # 三大功能面: 垂直面, 矢状面, 水平面
    plane_v = np.array([[S, 0, S], [-S, 0, S], [-S, 0, -S], [S, 0, -S]]) + center
    plane_s = np.array([[0, S, S], [0, -S, S], [0, -S, -S], [0, S, -S]]) + center
    plane_h = np.array([[S, S, 0], [-S, S, 0], [-S, -S, 0], [S, -S, 0]]) + center
    
    return {
        "center": c_pt, "cardinals": cardinals, "diagonals": diagonals,
        "icosahedron": icosahedron_pts, "edges": edges, "planes": [plane_v, plane_s, plane_h]
    }


def visualize_3d_joints_trajectory(
    npy_path, 
    frame_range=None,  # 可选参数：指定绘制的帧范围 (start_frame, end_frame)
    joint_names=['NOSE', 'LEFT_WRIST', 'RIGHT_WRIST'], 
    sample_rate=10, 
    view_key="Reconstructed_3D",
    draw_lines=True,
    draw_laban=False,      # [新增] 是否显示拉班27方位和二十面体
    laban_scale=0.9        # [新增] 拉班运动球体的半径(米)
):
    """
    绘制空间中指定关节的三维运动轨迹散点图，并支持叠加拉班空间几何。
    """
    # 检查关节名合法性
    for jn in joint_names:
        if jn not in POSE_LANDMARKS:
            print(f"❌ 错误: {jn} 不在关键点列表中")
            return

    print(f"正在加载 3D 数据: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    
    trajectories = {joint: {"X": [], "Y": [], "Z": [], "frames": []} for joint in joint_names}
    pelvis_X, pelvis_Y, pelvis_Z = [], [], [] # 存储骨盆用于计算拉班中心
    
    if frame_range:
        start_frame, end_frame = frame_range
        end_frame = min(end_frame, len(data) - 1)
    else:
        start_frame, end_frame = 0, len(data) - 1

    frames = sorted(list(data.keys()))
    for frame in frames[start_frame:end_frame + 1]:
        frame_data = data[frame]
        if view_key not in frame_data or len(frame_data[view_key]) == 0:
            continue
            
        world_kpts = frame_data[view_key][0]["world"]
        
        # 1. 收集目标关节的轨迹
        for joint in joint_names:
            j_idx = POSE_LANDMARKS.index(joint)
            X, Y, Z = world_kpts[j_idx]
            if X != 0 or Y != 0 or Z != 0:
                trajectories[joint]["X"].append(X)
                trajectories[joint]["Y"].append(Y)
                trajectories[joint]["Z"].append(Z)
                trajectories[joint]["frames"].append(frame)
                
        # 2. 收集骨盆中心数据 (如果需要画拉班)
        if draw_laban:
            lx, ly, lz = world_kpts[23] # LEFT_HIP
            rx, ry, rz = world_kpts[24] # RIGHT_HIP
            if lx != 0 or rx != 0:
                pelvis_X.append((lx + rx) / 2)
                pelvis_Y.append((ly + ry) / 2)
                pelvis_Z.append((lz + rz) / 2)

    # 准备画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap('tab10', len(joint_names))
    all_X, all_Y, all_Z = [], [], []

    # ==========================================
    # 绘制拉班二十面体和27方位点
    # ==========================================
    if draw_laban:
        if len(pelvis_X) > 0:
            laban_center = np.array([np.mean(pelvis_X), np.mean(pelvis_Y), np.mean(pelvis_Z)])
        else:
            laban_center = np.array([0, 0, 1.0]) # 默认后备中心
            
        laban_geom = generate_laban_kinesphere(laban_center, scale=laban_scale)
        
        # 画连线
        pts = laban_geom["icosahedron"]
        for (i, j) in laban_geom["edges"]:
            p1, p2 = pts[i], pts[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='black', alpha=0.5, lw=1.5, linestyle='--')
            
        # 画功能面
        # faces = Poly3DCollection(laban_geom["planes"], linewidths=1, edgecolors='k', alpha=0.02)
        # faces.set_facecolor('cyan')
        # ax.add_collection3d(faces)

        # 散点：27个方位
        ax.scatter(*laban_geom["center"][0], c='black', marker='X', s=100, label="Laban: Center")
        ax.scatter(laban_geom["cardinals"][:,0], laban_geom["cardinals"][:,1], laban_geom["cardinals"][:,2], 
                   c='blue', marker='s', s=30, label="Laban: 6 Cardinals")
        ax.scatter(laban_geom["diagonals"][:,0], laban_geom["diagonals"][:,1], laban_geom["diagonals"][:,2], 
                   c='green', marker='^', s=30, label="Laban: 8 Diagonals")
        ax.scatter(laban_geom["icosahedron"][:,0], laban_geom["icosahedron"][:,1], laban_geom["icosahedron"][:,2], 
                   c='red', marker='o', s=40, label="Laban: 12 Diametrals")
        
        # 将拉班边界加入 all_XYZ 以免被裁切
        bounds_min = laban_center - laban_scale * 1.1
        bounds_max = laban_center + laban_scale * 1.1
        all_X.extend([bounds_min[0], bounds_max[0]])
        all_Y.extend([bounds_min[1], bounds_max[1]])
        all_Z.extend([bounds_min[2], bounds_max[2]])

    # ==========================================
    # 绘制轨迹
    # ==========================================
    for idx, joint in enumerate(joint_names):
        Xs = np.array(trajectories[joint]["X"])
        Ys = np.array(trajectories[joint]["Y"])
        Zs = np.array(trajectories[joint]["Z"])
        frms = np.array(trajectories[joint]["frames"])
        
        if len(Xs) == 0:
            print(f"⚠️ 警告: 未找到 {joint} 的有效数据")
            continue
            
        all_X.extend(Xs)
        all_Y.extend(Ys)
        all_Z.extend(Zs)
        
        # 画线
        if draw_lines:
            ax.plot(Xs, Ys, Zs, color=colors(idx), alpha=0.3, linewidth=1.5, label=f"{joint} Path")
            
        # 散点和帧号
        sampled_indices = np.arange(0, len(Xs), sample_rate)
        s_Xs, s_Ys, s_Zs, s_frms = Xs[sampled_indices], Ys[sampled_indices], Zs[sampled_indices], frms[sampled_indices]
        
        ax.scatter(s_Xs, s_Ys, s_Zs, color=colors(idx), s=20,  depthshade=False,alpha=0.5)
        for x, y, z, f in zip(s_Xs, s_Ys, s_Zs, s_frms):
            ax.text(x, y, z + 0.02, str(f), color=colors(idx),alpha=0.6, fontsize=8)

    # ==========================================
    # 坐标轴与图例设置
    # ==========================================
    ax.set_xlabel("X (Width / Right) [m]")
    ax.set_ylabel("Y (Depth / Forward) [m]")
    ax.set_zlabel("Z (Height / Up) [m]")
    
    title_str = f"3D Joint Trajectories (Sample Rate: {sample_rate})"
    if frame_range:
        title_str += f" (Frames: {start_frame}-{end_frame})"
    if draw_laban: title_str += f"\n[Laban Kinesphere Scale: {laban_scale}m]"
    ax.set_title(title_str)
    
    # 强制 1:1:1 物理空间比例
    if len(all_X) > 0:
        max_range = np.array([
            max(all_X)-min(all_X), max(all_Y)-min(all_Y), max(all_Z)-min(all_Z)
        ]).max() / 2.0
        
        mid_x = (max(all_X) + min(all_X)) * 0.5
        mid_y = (max(all_Y) + min(all_Y)) * 0.5
        mid_z = (max(all_Z) + min(all_Z)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=20., azim=-45)
    
    # 优化图例：防止内容过多挡住图像，将其移至外部
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
    plt.tight_layout()
    plt.show()





# ==========================================
# 实际运行调用区
# ==========================================
if __name__ == "__main__":
    
    # 根据你之前估算出的像素焦距，除以图像宽度得到归一化焦距
    # 例如：如果像素焦距是 1587，图像宽是 2560，则 f_norm = 1587 / 2560 ≈ 0.620
    FOCAL_LENGTH_NORM_FRONT = 0.620
    FOCAL_LENGTH_NORM_LEFT = 0.620

    # 填入左侧相机的实际位置 (x0, y0)，单位：米
    # 表示相机在空间中心的左侧 x0 米，前方 y0 米处
    CAMERA_LEFT_POS = (5.27, 5.7) 
    CAMERA_HEIGHT = 1.35

    reconstruct_and_save_3d(
        input_npy_path="D:\\Desktop\\projects\\Dance-Movements-Analyzer\\mediapipe\\datasets\\251029whole_npy_final\\space\\freestyle-static-2\\merged_keypoints_ABC.npy",   # 你的双视角输入文件
        output_raw_npy_path="D:\\Desktop\\projects\\Dance-Movements-Analyzer\\mediapipe\\datasets\\joints_position_3d\\test_3d_poses_raw.npy", # 输出文件
        output_smooth_npy_path="D:\\Desktop\\projects\\Dance-Movements-Analyzer\\mediapipe\\datasets\\joints_position_3d\\test_3d_poses_smoothed.npy", # 输出文件
        front_view_key="A_keypoints.npy", 
        left_view_key="B_keypoints.npy",
        f1_norm=FOCAL_LENGTH_NORM_FRONT,
        f2_norm=FOCAL_LENGTH_NORM_LEFT,
        cam2_pos=CAMERA_LEFT_POS,
        cam_height=CAMERA_HEIGHT,
        smooth_window=15,  # 设置窗口大小：根据视频帧率调整，60fps设为21~31，30fps设为11~15效果较好
        smooth_poly=3      # 一般设为 3
    )

    visualize_3d_joints_trajectory(
        npy_path="D:\\Desktop\\projects\\Dance-Movements-Analyzer\\mediapipe\\datasets\\joints_position_3d\\test_3d_poses_smoothed.npy",
        # joint_names=['NOSE', 'LEFT_WRIST', 'LEFT_ANKLE','RIGHT_WRIST', 'RIGHT_ANKLE'], 
        frame_range=(50, 100),  # 可选：指定绘制的帧范围 (start_frame, end_frame)
        joint_names=['LEFT_SHOULDER','LEFT_WRIST'],
        sample_rate=10,  # 每 20 帧画一个点并标注帧号
        draw_lines=True,
        draw_laban=True,       # <--- 开启拉班绘制开关
        laban_scale=0.9        # <--- 可调整球体大小
    )