from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import math
import argparse

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from clip.clip import _transform
from timm.utils import accuracy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/duanzhenya/ood/gallop777/')))
import gallop.lib as lib
import gallop.vlprompt.tools as vlp_tools
import gallop.datasets.tools as dts_tools
from gallop.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from gallop.vlprompt import GalLoP
from gallop.vlprompt.tools import GlobalLocalLoss
from gallop.vlprompt.region_feature import get_region_features, RegionFeatureProjector, select_regions, augment_region
from gallop.vlprompt.gallop import select_regions_by_text_similarity
from gallop.vlprompt.grounded_sam2_utils import (
    load_groundingdino_model, 
    get_grounded_sam2_features,
    get_grounded_sam2_features_batch,
    grounded_sam2_pipeline,
    grounded_sam2_pipeline_batch
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# GroundingDINO imports - 使用直接路径导入
import sys
import os
gsam2_path = os.path.join(os.path.dirname(__file__), "..", "Grounded-SAM-2-main")
if gsam2_path not in sys.path:
    sys.path.append(gsam2_path)

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30028"

NoneType = Type[None]


def train_one_epoch(
    model: GalLoP,
    train_loader: DataLoader,
    loss_fn: GlobalLocalLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
    sam2_model=None,
    sam_proj=None,
    groundingdino_model=None,
) -> lib.DictAverage:
    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")

    class_names = train_loader.dataset.all_names

    if not args.learn_global_prompt and not args.learn_local_prompts:
        with torch.no_grad(), autocast(enabled=args.use_fp16):
            text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = local_text_features = None

    model.train()
    optimizer.zero_grad()
    track_loader = lib.track(train_loader, f"Epoch {epoch} / {args.max_epoch}")
    for i, batch in enumerate(track_loader):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        prompts = batch["prompt"] if "prompt" in batch else [class_names[t.item()] for t in targets]
        
        # 使用 Grounded SAM2 或原来的 SAM2
        if args.use_groundingdino and groundingdino_model is not None:
            # Grounded SAM2 pipeline - 批量版本，支持多卡并行
            region_features_list = get_grounded_sam2_features_batch(
                images, prompts, groundingdino_model, sam2_model, sam_proj,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold
            )
            # 将list转换为tensor，处理不同数量的region
            max_regions = max([len(feats) for feats in region_features_list]) if region_features_list else 1
            region_features = torch.zeros(images.size(0), max_regions, region_features_list[0][0].shape[-1]).cuda()
            for b, feats in enumerate(region_features_list):
                if len(feats) > 0:
                    region_features[b, :len(feats)] = torch.stack(feats)
        else:
            # 原来的 SAM2 patch embedding
            region_features = get_region_features(images, sam2_model, sam_proj)
        
        # 使用模型编码文本特征
        text_features, _ = model.encode_text(prompts)
        k = max(1, region_features.shape[1] // 5)
        pos_indices, neg_indices = select_regions_by_text_similarity(region_features, text_features, k)
        pos_region_features = torch.stack([
            region_features[b][pos_indices[b]] for b in range(images.size(0))
        ])
        neg_region_features = torch.stack([
            region_features[b][neg_indices[b]] for b in range(images.size(0))
        ])
        with autocast(enabled=args.use_fp16):
            global_logits, local_logits, neg_local_logits = model(
                images,
                class_names,
                text_features=None,
                local_text_features=None,
                pos_region_features=pos_region_features,
                neg_region_features=neg_region_features,
            )
            # 获取image_features和text_features用于prompt masking
            image_features, _ = model.encode_image_and_proj(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if hasattr(model, 'global_prompt') and model.learn_global_prompt:
                text_features_mask = model._prompt_features(model.global_prompt)
            else:
                text_features_mask, _ = model.encode_text(class_names)
            text_features_mask = text_features_mask / text_features_mask.norm(dim=-1, keepdim=True)
            loss = loss_fn(global_logits, local_logits, targets, model.logit_scale.exp(), neg_local_logits=neg_local_logits, image_features=image_features, text_features=text_features_mask)

        fp16_scaler.scale(loss).backward()
        track_loader.set_postfix({"gpu": torch.cuda.max_memory_allocated() / 1024**3})
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        optimizer.zero_grad()

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        topk = accuracy(gl_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        meter.update(
            {
                "loss": loss.detach().item(),
                "top1": topk[0],
                "top1_global": global_topk[0],
            },
            images.size(0),
        )

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_local": local_topk[0],
                },
                images.size(0),
            )

    progress.display_summary()

    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    model: GalLoP,
    val_loader: DataLoader,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]
    for batch in lib.track(val_loader, f"Evaluating on {dataset_name}"):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast(enabled=args.use_fp16):
            global_logits, local_logits, neg_local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
            # 撤销推理时的prompt masking，不再获取image_features和text_features_mask
            if return_scores:
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        if neg_local_logits is not None:
            neg_local_logits_mean = neg_local_logits.mean(dim=-1)
            neg_local_probs = torch.softmax(model.logit_scale.exp() * neg_local_logits_mean, dim=-1)
            top1_neg_local = accuracy(neg_local_probs, targets, topk=(1,))[0]
        else:
            top1_neg_local = None

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            topk = accuracy(gl_probs, targets, topk=(1,))
            logs = {
                "top1": topk[0],
                "top1_global": global_topk[0],
                "top1_local": local_topk[0],
            }
            if top1_neg_local is not None:
                logs["top1_neg_local"] = top1_neg_local
        else:
            logs = {
                "top1": global_topk[0],
                "top1_global": global_topk[0],
            }
            if top1_neg_local is not None:
                logs["top1_neg_local"] = top1_neg_local

        meter.update(logs, images.size(0))

    model.train(mode)
    return meter, test_scores


@torch.no_grad()
def evaluate_ood(
    model: GalLoP,
    val_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    test_scores: Optional[np.ndarray] = None,
) -> lib.DictAverage:
    metrics = defaultdict(dict)

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    if test_scores is None:
        test_scores = np.zeros(len(val_loader.dataset))
        for batch in lib.track(val_loader, "Computing ood scores for Test"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(enabled=args.use_fp16):
                global_logits, local_logits, neg_local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                # 撤销推理时的prompt masking，不再获取image_features和text_features_mask
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = np.zeros(len(ood_loader.dataset))
        for batch in lib.track(ood_loader, f"Computing ood scores for {ood_name}"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(args.use_fp16):
                global_logits, local_logits, neg_local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                # 撤销推理时的prompt masking，不再获取image_features和text_features_mask
                ood_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        metrics[ood_name]["fpr95"] = lib.get_fpr(test_scores, ood_scores)
        metrics[ood_name]["auroc"] = lib.get_auroc(test_scores, ood_scores)

    model.train(mode)
    return metrics


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser("Learning prompts for CLIP with local and global features")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="/home/duanzhenya/ood/gallop777/gallop/DATA", type=str)
    parser.add_argument("--save_dir", default="./results/result", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_shots", default=16, type=int, help="Number of shots by class. -1 means the whole dataset")
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--topk", default=[5, 10, 15, 20], type=int, nargs="+")
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompt", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--ood_method", default="GL-MCM", type=str)
    parser.add_argument("--ood_temp_scale", default=1.0, type=float)

    parser.add_argument("--clip_name", required=True, choices=clip_model_names, type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    parser.add_argument("--sam2_config", type=str, required=True, default="/home/duanzhenya/ood/gallop777/Grounded-SAM-2-main/sam2/configs/sam2/sam2_hiera_t.yaml",help="Path to SAM2 config yaml")
    parser.add_argument("--sam2_ckpt", type=str, required=True,default="/home/duanzhenya/ood/gallop777/Grounded-SAM-2-main/checkpoints/sam2_hiera_tiny.pt", help="Path to SAM2 checkpoint")
    
    # GroundingDINO parameters
    parser.add_argument("--groundingdino_config", type=str, default="/home/duanzhenya/ood/gallop777/Grounded-SAM-2-main/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to GroundingDINO config")
    parser.add_argument("--groundingdino_ckpt", type=str, default="/home/duanzhenya/ood/gallop777/Grounded-SAM-2-main/gdino_checkpoints/groundingdino_swint_ogc.pth",
        help="Path to GroundingDINO checkpoint")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="GroundingDINO box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text confidence threshold")
    parser.add_argument("--use_groundingdino", type=lib.boolean_flags, default=True, help="Whether to use GroundingDINO for guided segmentation")

    parser.add_argument("--region_select_mode", type=str, default="combined",
        choices=["all", "most_similar", "least_similar", "most_confident", "least_confident", "combined"],
        help="区域筛选方式，用于消融实验")
    parser.add_argument("--region_augment_mode", type=str, default="occlude",
        choices=["none", "occlude", "jitter"], help="区域增强方式")
    parser.add_argument("--region_topk", type=int, default=1, help="每种方式选几个区域")

    args = parser.parse_args()

    lib.setup_logger()
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)

    args.eval_domains = args.eval_domains and (args.dataset_name == "imagenet")
    args.eval_ood = args.eval_ood and (args.dataset_name == "imagenet")

    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up Imagenet dataset train
    train_dataset, val_dataset, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform)
    template = "A photo of a {}" if (args.learn_global_prompt or args.learn_local_prompts) else template

    train_dataset = dts_tools.create_few_shots_dataset(train_dataset, args.num_shots, seed=args.seed)
    lib.LOGGER.info("Using template: " + template.format("<class_name>"))

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(val_dataset, batch_size=args.inference_batch_size)

    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)



    # Setting-up model
    model = GalLoP(
        clip_name=args.clip_name,
        use_local_features=args.use_local_features,
        checkpointing_segments=args.checkpointing_segments,
        template=template,
        learn_local_proj=args.learn_local_proj,
        learn_local_prompts=args.learn_local_prompts,
        learn_global_prompt=args.learn_global_prompt,
        class_names=train_dataset.all_names,
        n_global_prompts=args.n_global_prompts,
        n_local_prompts=args.n_local_prompts,
        prompts_batch_size=args.prompts_batch_size,
        ood_method=args.ood_method,
        ood_temp_scale=args.ood_temp_scale,
        topk=args.topk,
        parallel_text_encoder=args.parallel_text_encoder,
        parallel_vision_encoder=args.parallel_vision_encoder,
    )

    model.initialize_prompt()

    # eventually load pre-trained prompts
    lib.load_checkpoint(model, args.checkpoint_path)

    model.freeze_clip()
    model = model.cuda()

    # 加载 GroundingDINO 模型
    groundingdino_model = None
    if args.use_groundingdino:
        groundingdino_model = load_groundingdino_model(
            args.groundingdino_config, 
            args.groundingdino_ckpt, 
            device="cuda"
        )
        
        if groundingdino_model is not None:
            # 多卡并行
            if torch.cuda.device_count() > 1:
                groundingdino_model = torch.nn.DataParallel(groundingdino_model)
            # 冻结参数
            groundingdino_model.eval()
            for param in groundingdino_model.parameters():
                param.requires_grad = False
            print("✅ GroundingDINO 模型加载成功")
        else:
            print("⚠️  GroundingDINO 模型加载失败，将仅使用 SAM2")
            args.use_groundingdino = False
    
    # 加载 SAM2 模型
    # 检查配置文件路径
    sam2_config_path = args.sam2_config
    if not os.path.isabs(sam2_config_path):
        # 如果是相对路径，转换为绝对路径
        sam2_config_path = os.path.abspath(sam2_config_path)
    
    print(f"🔧 使用 SAM2 配置文件: {sam2_config_path}")
    sam2_model = build_sam2(sam2_config_path, args.sam2_ckpt, device="cuda")
    if torch.cuda.device_count() > 1:
        sam2_model = torch.nn.DataParallel(sam2_model)
    # 冻结 SAM2
    for param in sam2_model.parameters():
        param.requires_grad = False
    sam2_model.eval()
    for m in sam2_model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.train = lambda _: None
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # 自动推断sam特征维度和clip特征维度，初始化sam_proj
    # 取一张图片推理一次，获得region_features.shape[-1]和clip特征维度
    dummy_img = torch.randn(1, 3, 224, 224).cuda()
    dummy_prompts = ["a photo of a cat"]
    
    with torch.no_grad():
        if args.use_groundingdino and groundingdino_model is not None:
            # 使用 Grounded SAM2 - 批量版本
            region_features_list = get_grounded_sam2_features_batch(
                dummy_img, dummy_prompts, groundingdino_model, sam2_predictor,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold
            )
            # 取第一个batch的第一个region特征
            if len(region_features_list) > 0 and len(region_features_list[0]) > 0:
                sam_dim = region_features_list[0][0].shape[-1]
            else:
                # 如果没有检测到区域，用SAM2的patch embedding
                image_features = sam2_model.get_image_embedding(dummy_img)
                sam_dim = image_features.shape[1]
        else:
            # 使用原来的SAM2 patch embedding
            region_features = get_region_features(dummy_img, sam2_model, None)  # sam_proj 还没有初始化
            sam_dim = region_features.shape[-1]
        
        text_features, _ = model.encode_text(["a photo of a cat"])
        clip_dim = text_features.shape[-1]
    
    sam_proj = RegionFeatureProjector(sam_dim, clip_dim).cuda()

    # setting-up loss
    loss_fn = GlobalLocalLoss(
        use_global_loss=args.use_global_loss,
        use_local_loss=args.use_local_loss,
        use_neg_local_loss=True,
        topk=args.topk,
        global_dropout_p=args.global_dropout_p,
    )

    # Setting-up optimizer
    optimizer = vlp_tools.get_optimizer(args.optimizer, list(model.parameters()) + list(sam_proj.parameters()), args.lr_init, args.weight_decay, args.momentum)

    # Setting-up scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    # Setting-up GradScaler for amp
    fp16_scaler = GradScaler(enabled=args.use_fp16)

    # Training loop
    for epoch in range(args.max_epoch):
        if not args.eval_only:
            assert args.use_local_loss or args.use_global_loss or args.learn_local_prompts or args.learn_global_prompt, "At least one of use_local_loss or use_global_loss or learn_local_prompts or learn_global_prompt must be True"
            train_meter = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
                sam2_model=sam2_model,
                sam_proj=sam_proj,
                groundingdino_model=groundingdino_model,
            )

            lib.save_checkpoint(args.save_dir, epoch, model, optimizer, lr_scheduler, fp16_scaler, train_meter, args)

        if ((epoch % args.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == args.max_epoch) or args.eval_only:
            lib.LOGGER.info("Evaluation")
            val_meter, test_scores = evaluate(model, val_loader, args, return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)))
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))

            if args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)):
                ood_metrics = evaluate_ood(model, val_loader, ood_loaders, args, test_scores=test_scores)
                lib.LOGGER.info(f"OOD Evaluation metrics with temperature scale {args.ood_temp_scale} (FPR95 / AUROC): ")
                lib.log_ood_metrics(ood_metrics)

            if args.eval_domains and (args.eval_only or (epoch + 1 == args.max_epoch)):
                metrics = {}
                for domain_name, domain_loader in domains_loaders.items():
                    metrics[domain_name], _ = evaluate(model, domain_loader, args)
                    lib.LOGGER.info(f"Evaluation metrics for {domain_name}: " + " ".join([" *"] + metrics[domain_name].summary()))
                avg_top1 = np.mean([metrics[domain_name].avg["top1"] for domain_name in domains_loaders.keys()])
                lib.LOGGER.info(f"Average evaluation metrics for domains: * top1: {avg_top1: .3f}")

            if args.eval_only:
                break
