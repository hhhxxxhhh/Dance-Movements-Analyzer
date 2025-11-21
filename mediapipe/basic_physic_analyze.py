import numpy as np

# 替换为你的.npy文件路径（相对路径或绝对路径均可）
npy_file_path = r"D:\Desktop\projects\DanceAnalyze\datasets\251029whole_npy\emotion\emotion-angry-1\A_keypoints.npy"  # 示例：相对路径，如"251029npy/A.npy"
# 也可以是绝对路径，如 Windows 下的 r"C:\path\to\your_file.npy" 或 Linux/Mac 下的 "/path/to/your_file.npy"

# 加载.npy文件到变量
data = np.load(npy_file_path, allow_pickle=True)

single_dict = data.item()
#single_dict['frame_1800'][0]['world'] to get xyz of a single frame