import torch
import clip
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CLIP 模型初始化 ----------------
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------- LMA 文本提示 ----------------

LMA_PROMPTS = {
    "Body": {
        "arms": ["the dancer nearly doesn't move arms", "the dancer moves arms gracefully"],
        "legs": ["the dancer nearly doesn't move legs", "the dancer moves legs energetically"],
        "torso": ["the dancer nearly doesn't move torso", "the dancer moves torso subtly"]
    },
    "Effort": {
        "Time": ["the dancer moves slowly", "the dancer moves quickly"],
        "Weight": ["the dancer moves lightly", "the dancer moves strongly"],
        "Flow": ["the dancer moves continuously", "the dancer moves abruptly"],
        "Space": ["the dancer moves with narrow attention", "the dancer moves with wide attention"]
    },
    "Shape": {
        "rising": ["the dancer descends", "the dancer rises upward"],
        "widening": ["the dancer contracts sideways", "the dancer expands sideways"],
        "kinesphere": ["the dancer moves in open space", "the dancer moves in contained space"]
    },
    "Space": {
        "dominant_direction": ["the dancer moves without clear direction", "the dancer moves in a clear direction"],
        "concentration": ["the dancer moves broadly", "the dancer concentrates movement in a small area"],
        "non_directional": ["the dancer moves in a clear direction", "the dancer moves without clear direction"]
    }
}

# 预编码文本
TEXT_FEATURES = {}
for dim, prompts in LMA_PROMPTS.items():
    TEXT_FEATURES[dim] = {}
    for label, desc_list in prompts.items():
        # desc_list = [low_desc, high_desc]
        text_tokens = clip.tokenize(desc_list).to(device)
        with torch.no_grad():
            feats = model.encode_text(text_tokens)  # (2, feature_dim)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        TEXT_FEATURES[dim][label] = feats  # 保存为 tensor(2, feature_dim)

# ---------------- 视频帧提取 ----------------
def video_to_frames(video_path, resize=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames

# ---------------- CLIP 映射到 LMA ----------------
# def clip_lma_analysis(frames, segment_len=30):
#     """
#     frames: list of RGB numpy arrays
#     segment_len: 每个 segment 帧数
#     输出每个 segment 每个维度两个反义描述的平均相似度
#     """
#     results = []
#     T = len(frames)
#     for start in range(0, T, segment_len):
#         end = min(start + segment_len, T)
#         seg_frames = frames[start:end]

#         # 编码帧
#         frame_feats = []
#         for f in seg_frames:
#             img_input = preprocess(Image.fromarray(f)).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 feat = model.encode_image(img_input)
#                 feat = feat / feat.norm(dim=-1, keepdim=True)
#                 frame_feats.append(feat)
#         frame_feats = torch.cat(frame_feats, dim=0)  # (seg_len, feature_dim)

#         seg_result = {}
#         for dim, text_feats in TEXT_FEATURES.items():
#             dim_result = {}
#             for label, t_feat in text_feats.items():
#                 # t_feat 是 (2, feature_dim) 各反义描述
#                 sim_scores = []
#                 for desc_feat in t_feat:
#                     sim = (frame_feats @ desc_feat.unsqueeze(-1)).squeeze(-1)
#                     sim_scores.append(sim.mean().item())
#                 dim_result[label] = sim_scores  # [low, high]
#             # 选择得分高的作为 dominant
#             dominant_label = max(dim_result, key=lambda k: max(dim_result[k]))
#             seg_result[dim] = {
#                 "scores": dim_result,          # {label: [low_sim, high_sim]}
#                 "dominant_label": dominant_label
#             }
#         results.append(seg_result)
#     return results

def clip_lma_analysis(frames, segment_len=30):
    """
    frames: list of RGB numpy arrays
    segment_len: 每个 segment 帧数
    输出每个 segment 每个维度反义描述中得分更高的那个描述
    """
    results = []
    T = len(frames)
    for start in range(0, T, segment_len):
        end = min(start + segment_len, T)
        seg_frames = frames[start:end]

        # 编码帧
        frame_feats = []
        for f in seg_frames:
            img_input = preprocess(Image.fromarray(f)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                frame_feats.append(feat)
        frame_feats = torch.cat(frame_feats, dim=0)  # (seg_len, feature_dim)

        seg_result = {}
        for dim, text_feats in TEXT_FEATURES.items():
            dim_result = {}
            for label, t_feat in text_feats.items():
                # t_feat 是 (2, feature_dim) 各反义描述
                sim_scores = []
                for desc_feat in t_feat:
                    sim = (frame_feats @ desc_feat.unsqueeze(-1)).squeeze(-1)
                    sim_scores.append(sim.mean().item())
                # 选择分数更高的描述
                max_idx = int(np.argmax(sim_scores))
                dim_result[label] = {
                    "best_score": sim_scores[max_idx],
                    "best_desc_index": max_idx  # 0=low_desc, 1=high_desc
                }
            # 选择得分最高的 label 作为 dominant
            dominant_label = max(dim_result, key=lambda k: dim_result[k]["best_score"])
            seg_result[dim] = {
                "scores": dim_result,          # {label: {"best_score":..., "best_desc_index":0/1}}
                "dominant_label": dominant_label
            }
        results.append(seg_result)
    return results


# ---------------- 示例 ----------------
if __name__ == "__main__":
    video_path = "D:\\Desktop\\projects\\Dance-Movements-Analyzer\\test_video.mp4"
    frames = video_to_frames(video_path)
    segment_results = clip_lma_analysis(frames, segment_len=60)

    # 打印前两个 segment
    # for i, seg in enumerate(segment_results):
    #     print(f"Segment {i+1}")
    #     for dim, info in seg.items():
    #         scores_fmt = {lbl: [f"{s:.5f}" for s in sims] for lbl, sims in info['scores'].items()}
    #         print(f"  {dim}: dominant={info['dominant_label']}, scores={scores_fmt}")
    #     print("---------")

    for i, seg in enumerate(segment_results):
        print(f"Segment {i+1}")
        for dim, info in seg.items():
            scores_fmt = {lbl: f"{v['best_score']:.5f} ({v['best_desc_index']})" 
                        for lbl, v in info['scores'].items()}
            print(f"  {dim}: dominant={info['dominant_label']}, best_scores={scores_fmt}")
        print("---------")
