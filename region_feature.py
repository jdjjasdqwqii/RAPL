import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 可学习的SAM特征投影层
class RegionFeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        # x: [B, N, in_dim] or [N, in_dim]
        return self.proj(x)

# 区域筛选函数
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def select_regions(region_feats, global_feat, region_scores, mode="all", topk=1):
    N = region_feats.shape[0]
    indices = []
    if mode == "all":
        indices = list(range(N))
    else:
        if "similar" in mode:
            sims = [cosine_similarity(region_feats[i], global_feat) for i in range(N)]
            sims = np.array(sims)
            if mode == "most_similar":
                indices = sims.argsort()[-topk:][::-1]
            elif mode == "least_similar":
                indices = sims.argsort()[:topk]
        if "confident" in mode:
            scores = region_scores if isinstance(region_scores, np.ndarray) else region_scores.cpu().numpy()
            if mode == "most_confident":
                indices = scores.argsort()[-topk:][::-1]
            elif mode == "least_confident":
                indices = scores.argsort()[:topk]
        if mode == "combined":
            sims = [cosine_similarity(region_feats[i], global_feat) for i in range(N)]
            sims = np.array(sims)
            scores = region_scores if isinstance(region_scores, np.ndarray) else region_scores.cpu().numpy()
            idxs = set()
            idxs.add(int(sims.argmax()))
            idxs.add(int(sims.argmin()))
            idxs.add(int(scores.argmax()))
            idxs.add(int(scores.argmin()))
            indices = list(idxs)
    return indices

# 区域增强函数
def augment_region(region, mask, mode="occlude"):
    if mode == "occlude":
        h, w, _ = region.shape
        x0 = np.random.randint(0, w // 2)
        y0 = np.random.randint(0, h // 2)
        x1 = np.random.randint(w // 2, w)
        y1 = np.random.randint(h // 2, h)
        region_aug = region.copy()
        region_aug[y0:y1, x0:x1, :] = 0
        return region_aug, mask
    elif mode == "jitter":
        region_aug = region.astype(np.float32)
        region_aug *= np.random.uniform(0.8, 1.2, size=(1, 1, 3))
        region_aug = np.clip(region_aug, 0, 255).astype(np.uint8)
        return region_aug, mask
    else:
        return region, mask

# sam2_model: SAM2模型实例（Grounded-SAM-2-main）
def get_region_features(image, sam2_model, sam_proj=None):
    """
    输入:
        image: torch.Tensor, [C, H, W] 或 [B, C, H, W]
        sam2_model: 已加载的 SAM2 模型实例
        sam_proj: 可选，RegionFeatureProjector实例
    输出:
        region_features: torch.Tensor, [num_regions, dim] 或 [B, num_regions, dim]
    """
    sam2_model.eval()
    with torch.no_grad():
        # 确保输入是 torch.Tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        else:
            raise ValueError("image must be torch.Tensor")
        
        # 直接使用 SAM2 模型获取特征
        feats = sam2_model.get_image_embedding(image)  # [B, C, H, W]
        B, C, H, W = feats.shape
        region_features = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        
        # 投影到CLIP空间
        if sam_proj is not None:
            region_features = sam_proj(region_features)
        
        return region_features 

# 区域选择可视化
# image: numpy array (H, W, 3)
# region_indices: list of int, 被选中的区域索引
# region_shape: (h, w), patch大小
# grid_shape: (H, W), patch网格数
# save_path: 可选，保存路径

def visualize_selected_regions(image, region_indices, region_shape, grid_shape, save_path=None, title=None):
    """
    在原图上高亮显示被选中的patch/区域
    image: HWC, uint8 or float
    region_indices: list of int
    region_shape: (h, w)
    grid_shape: (H, W)
    save_path: str or None
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    h, w = region_shape
    grid_H, grid_W = grid_shape
    for idx in region_indices:
        row = idx // grid_W
        col = idx % grid_W
        rect = patches.Rectangle((col * w, row * h), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show() 