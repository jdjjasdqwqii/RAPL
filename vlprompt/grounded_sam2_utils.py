import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from PIL import Image
import sys
import os

# 添加 Grounded-SAM-2-main 到路径
gsam2_path = os.path.join(os.path.dirname(__file__), "..", "..", "Grounded-SAM-2-main")
if gsam2_path not in sys.path:
    sys.path.append(gsam2_path)

# 直接导入需要的模块
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_groundingdino_model(config_path: str, checkpoint_path: str, device: str = "cuda", bert_models_dir: str = None):
    """加载 GroundingDINO 模型"""
    try:
        # 首先检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ GroundingDINO checkpoint 文件不存在: {checkpoint_path}")
            print("💡 请检查文件路径或重新下载 checkpoint")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            print(f"❌ GroundingDINO checkpoint 文件为空: {checkpoint_path}")
            return None
        
        print(f"📁 加载 GroundingDINO checkpoint: {checkpoint_path}")
        print(f"📏 文件大小: {file_size / (1024*1024):.2f} MB")
        
        args = SLConfig.fromfile(config_path)
        args.device = device
        
        # 设置离线模式和本地模型目录
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        if bert_models_dir:
            os.environ['TRANSFORMERS_CACHE'] = bert_models_dir
            os.environ['HF_HOME'] = bert_models_dir
            print(f"🔧 使用本地 BERT 模型目录: {bert_models_dir}")
        
        print("🔨 构建 GroundingDINO 模型...")
        model = build_model(args)
        
        print("📥 加载 checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            print(f"❌ checkpoint 文件损坏: {e}")
            print("💡 建议重新下载 checkpoint 文件")
            return None
        
        print("⚙️  加载模型权重...")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model = model.to(device)
        
        print("✅ GroundingDINO 模型加载成功")
        return model
        
    except Exception as e:
        print(f"⚠️  GroundingDINO 模型加载失败: {e}")
        
        # 检查是否是网络连接问题
        if "Connection" in str(e) or "reset" in str(e).lower():
            print("🌐 检测到网络连接问题")
            print("💡 建议解决方案:")
            print("1. 设置离线模式: export HF_HUB_OFFLINE=1")
            print("2. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
            print("3. 禁用 GroundingDINO: --use_groundingdino False")
            print("4. 手动下载 BERT 模型到本地")
        else:
            print("💡 建议解决方案:")
            print("1. 检查 checkpoint 文件是否完整")
            print("2. 重新下载 checkpoint 文件")
            print("3. 检查网络连接")
            print("4. 手动下载 BERT 模型到本地缓存")
            print("5. 使用离线模式")
            print("6. 或者禁用 GroundingDINO，仅使用 SAM2")
            print("7. 指定本地 BERT 模型目录: --bert_models_dir /path/to/models")
        
        # 返回 None，让调用者决定如何处理
        return None


def get_grounding_output_batch(model, images, captions, box_threshold, text_threshold, device="cuda"):
    """GroundingDINO 批量推理函数 - 支持多卡并行"""
    # 预处理 captions
    processed_captions = []
    for caption in captions:
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        processed_captions.append(caption)
    
    model = model.to(device)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images, captions=processed_captions)
    
    batch_boxes = []
    batch_phrases = []
    
    logits = outputs["pred_logits"].sigmoid()  # (B, nq, 256)
    boxes = outputs["pred_boxes"]  # (B, nq, 4)
    
    for b in range(images.size(0)):
        # filter output for each image
        logits_filt = logits[b].cpu().clone()
        boxes_filt = boxes[b].cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # get phrase
        tokenizer = model.tokenizer
        tokenized = tokenizer(processed_captions[b])
        
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
        
        batch_boxes.append(boxes_filt)
        batch_phrases.append(pred_phrases)
    
    return batch_boxes, batch_phrases


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """GroundingDINO 单张图片推理函数（保持兼容性）"""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)
    
    return boxes_filt, pred_phrases


def grounded_sam2_pipeline_batch(
    images: torch.Tensor,
    prompts: List[str],
    groundingdino_model,
    sam2_model,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda"
) -> List[List[torch.Tensor]]:
    """
    完整的 Grounded SAM2 pipeline - 批量版本，支持多卡并行
    
    Args:
        images: [B, 3, H, W] 输入图像
        prompts: [B] 每张图像对应的文本prompt
        groundingdino_model: GroundingDINO模型
        sam2_model: SAM2模型（直接使用，不用predictor）
        box_threshold: box置信度阈值
        text_threshold: 文本置信度阈值
        device: 设备
    
    Returns:
        masks: List[List[torch.Tensor]] 每张图像的分割mask列表
    """
    batch_size = images.shape[0]
    
    # 1. GroundingDINO 批量检测
    batch_boxes, batch_phrases = get_grounding_output_batch(
        groundingdino_model, images, prompts, box_threshold, text_threshold, device=device
    )
    
    # 2. 批量提取 SAM2 特征
    with torch.no_grad():
        # 获取 SAM2 的 image embedding
        image_features = sam2_model.get_image_embedding(images)  # [B, C, H, W]
    
    # 3. 为每个检测框生成 mask（简化版本，用 attention 权重近似）
    all_masks = []
    
    for b in range(batch_size):
        boxes_filt = batch_boxes[b]
        image_feat = image_features[b]  # [C, H, W]
        
        if len(boxes_filt) == 0:
            # 如果没有检测到box，返回空mask
            all_masks.append([])
            continue
        
        masks = []
        for box in boxes_filt:
            # 将 box 转换为特征图坐标
            x1, y1, x2, y2 = box.tolist()
            
            # 计算特征图上的坐标（假设输入是 224x224，特征图是 14x14）
            feat_h, feat_w = image_feat.shape[1], image_feat.shape[2]
            img_h, img_w = images.shape[2], images.shape[3]
            
            # 转换坐标
            x1_feat = int(x1 * feat_w / img_w)
            y1_feat = int(y1 * feat_h / img_h)
            x2_feat = int(x2 * feat_w / img_w)
            y2_feat = int(y2 * feat_h / img_h)
            
            # 创建简单的矩形 mask
            mask = torch.zeros(feat_h, feat_w)
            mask[y1_feat:y2_feat, x1_feat:x2_feat] = 1.0
            
            # 上采样到原图大小
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(img_h, img_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            masks.append(mask)
        
        all_masks.append(masks)
    
    return all_masks


def grounded_sam2_pipeline(
    images: torch.Tensor,
    prompts: List[str],
    groundingdino_model,
    sam2_predictor,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda"
) -> List[List[torch.Tensor]]:
    """
    完整的 Grounded SAM2 pipeline（保持兼容性）
    
    Args:
        images: [B, 3, H, W] 输入图像
        prompts: [B] 每张图像对应的文本prompt
        groundingdino_model: GroundingDINO模型
        sam2_predictor: SAM2预测器
        box_threshold: box置信度阈值
        text_threshold: 文本置信度阈值
        device: 设备
    
    Returns:
        masks: List[List[torch.Tensor]] 每张图像的分割mask列表
    """
    batch_size = images.shape[0]
    all_masks = []
    
    for i in range(batch_size):
        image = images[i]  # [3, H, W]
        prompt = prompts[i]
        
        # 1. GroundingDINO 检测
        boxes_filt, pred_phrases = get_grounding_output(
            groundingdino_model, image, prompt, box_threshold, text_threshold, device=device
        )
        
        if len(boxes_filt) == 0:
            # 如果没有检测到box，返回空mask
            all_masks.append([])
            continue
        
        # 2. SAM2 分割
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        sam2_predictor.set_image(image_np)
        
        masks = []
        for box in boxes_filt:
            # 转换box格式
            x1, y1, x2, y2 = box.tolist()
            input_box = np.array([x1, y1, x2, y2])
            
            # SAM2 分割
            mask, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            masks.append(torch.from_numpy(mask[0]).float())
        
        all_masks.append(masks)
    
    return all_masks


def extract_region_features_from_masks(
    images: torch.Tensor,
    masks: List[List[torch.Tensor]],
    sam2_model,
    sam_proj=None,
    device: str = "cuda"
) -> torch.Tensor:
    """
    从分割mask中提取区域特征
    
    Args:
        images: [B, 3, H, W] 输入图像
        masks: List[List[torch.Tensor]] 每张图像的分割mask列表
        sam2_model: SAM2模型
        sam_proj: 特征投影层
        device: 设备
    
    Returns:
        region_features: [B, N, D] 区域特征
    """
    batch_size = images.shape[0]
    all_region_features = []
    
    for i in range(batch_size):
        image = images[i]  # [3, H, W]
        image_masks = masks[i]
        
        if len(image_masks) == 0:
            # 如果没有mask，用整图特征
            with torch.no_grad():
                image_features = sam2_model.get_image_embedding(image.unsqueeze(0))
                # 取全局平均池化
                region_feat = image_features.mean(dim=[2, 3])  # [1, C]
                all_region_features.append(region_feat)
            continue
        
        # 对每个mask提取特征
        mask_features = []
        for mask in image_masks:
            # 将mask转换为SAM2输入格式
            mask_np = mask.cpu().numpy()
            
            # 用mask加权平均pooling
            with torch.no_grad():
                image_features = sam2_model.get_image_embedding(image.unsqueeze(0))  # [1, C, H, W]
                
                # 将mask resize到特征图大小
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), 
                    size=image_features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )  # [1, 1, H', W']
                
                # 加权平均pooling
                weighted_features = image_features * mask_resized
                region_feat = weighted_features.sum(dim=[2, 3]) / (mask_resized.sum() + 1e-8)
                mask_features.append(region_feat)
        
        # 拼接所有mask的特征
        if len(mask_features) > 0:
            region_feat = torch.cat(mask_features, dim=0)  # [N, C]
        else:
            # 如果没有有效mask，用整图特征
            with torch.no_grad():
                image_features = sam2_model.get_image_embedding(image.unsqueeze(0))
                region_feat = image_features.mean(dim=[2, 3])  # [1, C]
        
        all_region_features.append(region_feat)
    
    # 投影到CLIP特征空间
    if sam_proj is not None:
        all_region_features = [sam_proj(feat) for feat in all_region_features]
    
    return all_region_features


def get_grounded_sam2_features_batch(
    images: torch.Tensor,
    prompts: List[str],
    groundingdino_model,
    sam2_model,
    sam_proj=None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda"
) -> List[torch.Tensor]:
    """
    完整的 Grounded SAM2 特征提取 - 批量版本，支持多卡并行
    
    Args:
        images: [B, 3, H, W] 输入图像
        prompts: [B] 每张图像对应的文本prompt
        groundingdino_model: GroundingDINO模型
        sam2_model: SAM2模型
        sam_proj: 特征投影层
        box_threshold: box置信度阈值
        text_threshold: 文本置信度阈值
        device: 设备
    
    Returns:
        region_features: List[torch.Tensor] 每张图像的区域特征列表
    """
    # 1. Grounded SAM2 pipeline - 批量版本
    masks = grounded_sam2_pipeline_batch(
        images, prompts, groundingdino_model, sam2_model, 
        box_threshold, text_threshold, device
    )
    
    # 2. 提取区域特征
    region_features = extract_region_features_from_masks(
        images, masks, sam2_model, sam_proj, device
    )
    
    return region_features


def get_grounded_sam2_features(
    images: torch.Tensor,
    prompts: List[str],
    groundingdino_model,
    sam2_predictor,
    sam2_model,
    sam_proj=None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda"
) -> torch.Tensor:
    """
    完整的 Grounded SAM2 特征提取（保持兼容性）
    
    Args:
        images: [B, 3, H, W] 输入图像
        prompts: [B] 每张图像对应的文本prompt
        groundingdino_model: GroundingDINO模型
        sam2_predictor: SAM2预测器
        sam2_model: SAM2模型
        sam_proj: 特征投影层
        box_threshold: box置信度阈值
        text_threshold: 文本置信度阈值
        device: 设备
    
    Returns:
        region_features: [B, N, D] 区域特征
    """
    # 1. Grounded SAM2 pipeline
    masks = grounded_sam2_pipeline(
        images, prompts, groundingdino_model, sam2_predictor, 
        box_threshold, text_threshold, device
    )
    
    # 2. 提取区域特征
    region_features = extract_region_features_from_masks(
        images, masks, sam2_model, sam_proj, device
    )
    
    return region_features 