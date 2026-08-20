import numpy as np

def print_dict_tree(d, indent=0):
    """递归打印字典的层级树结构"""
    if isinstance(d, dict):
        for key, value in d.items():
            print("    " * indent + str(key))
            print_dict_tree(value, indent + 1)
    elif isinstance(d, (list, tuple)):
        for i, item in enumerate(d):
            print("    " * indent + f"[{i}]")
            print_dict_tree(item, indent + 1)
    # 如果是 ndarray 或其他类型就不展开，只打印层级
    else:
        return

if __name__ == "__main__":
    path ="D:\\Desktop\\projects\\Dance-Movements-Analyzer\\mediapipe\\datasets\\251029whole_npy_final\\emotion\\emotion-angry-1\\merged_keypoints_ABC.npy"
    data = np.load(path, allow_pickle=True).item()  # 确保是字典格式
    # print_dict_tree(data)
    print(data[1])
