from typing import Type, Any, Dict, Optional, List, Tuple

import math
import numpy as np

import clip
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
import torch.nn.functional as F

from clip import load as load_clip

import gallop.lib as lib
import gallop.vlprompt.tools as vlp_tools
from gallop.vlprompt.prompted_transformers import PromptedTransformer
from gallop.vlprompt.clip_local import ModifiedResNet, VisionTransformer, CLIP
# GroundingDINO imports - 使用直接路径导入
import sys
import os
gsam2_path = os.path.join(os.path.dirname(__file__), "..", "..", "Grounded-SAM-2-main")
if gsam2_path not in sys.path:
    sys.path.append(gsam2_path)

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
NoneType = Type[None]
KwargType = Dict[str, Any]
CLIP_NAME = {"clip_vit_b32": "ViT-B/32", "clip_vit_b16": "ViT-B/16", "clip_resnet50": "RN50", "clip_resnet101": "RN101"}


class GalLoP(CLIP):
    TRAINABLE_PARAMS: List[str] = []

    def __init__(
        self,
        clip_name: str,
        use_local_features: bool = True,
        checkpointing_segments: int = 8,
        template: str = "A photo of a {}",
        learn_local_proj: bool = True,
        learn_local_prompts: bool = True,
        learn_global_prompt: bool = True,
        class_names: List[str] = None,
        n_global_prompts: int = 1,
        n_local_prompts: int = 1,
        n_neg_prompts: int = 0,
        prompts_batch_size: int = math.inf,
        ood_method: str = "GL-MCM",
        ood_temp_scale: float = 1.0,
        topk: List[int] = [5, 10, 15, 20],
        parallel_text_encoder: bool = False,
        parallel_vision_encoder: bool = False,
    ) -> NoneType:
        self.model_name = "gallop_" + clip_name[5:]
        clip_model, _ = load_clip(CLIP_NAME[clip_name], device="cuda")

        clip_state_dict = clip_model.state_dict()
        clip_kwargs = lib.get_clip_hyperparams(clip_state_dict)
        clip_kwargs["return_local_features"] = use_local_features

        super().__init__(**clip_kwargs)
        self.clip_name = clip_name
        self.use_local_features = use_local_features
        self.learn_local_proj = learn_local_proj
        self.template = template[:-1] if template[-1] == "." else template
        self.learn_local_prompts = learn_local_prompts
        self.learn_global_prompt = learn_global_prompt
        self.class_names = class_names
        self.n_global_prompts = n_global_prompts
        self.n_local_prompts = n_local_prompts
        self.n_neg_prompts = n_neg_prompts
        self.prompts_batch_size = min(prompts_batch_size, self.n_global_prompts)
        self.ood_method = ood_method
        self.ood_temp_scale = ood_temp_scale
        self.topk = topk

        self.parallel_text_encoder = parallel_text_encoder
        self.parallel_vision_encoder = parallel_vision_encoder

        if isinstance(clip_kwargs["vision_layers"], (tuple, list)):
            self.visual = ModifiedResNet(
                layers=clip_kwargs["vision_layers"],
                output_dim=clip_kwargs["embed_dim"],
                heads=clip_kwargs["vision_width"] * 32 // 64,
                input_resolution=clip_kwargs["image_resolution"],
                width=clip_kwargs["vision_width"],
            )
            vision_dim = clip_kwargs["embed_dim"]
        else:
            self.visual = VisionTransformer(
                input_resolution=clip_kwargs["image_resolution"],
                patch_size=clip_kwargs["vision_patch_size"],
                width=clip_kwargs["vision_width"],
                layers=clip_kwargs["vision_layers"],
                heads=clip_kwargs["vision_width"] // 64,
                output_dim=clip_kwargs["embed_dim"],
            )
            vision_dim = clip_kwargs["vision_width"]

        self.transformer = PromptedTransformer(
            width=clip_kwargs["transformer_width"],
            layers=clip_kwargs["transformer_layers"],
            heads=clip_kwargs["transformer_heads"],
            attn_mask=self.build_attention_mask(),
            segments=checkpointing_segments,
        )

        if self.learn_global_prompt or self.learn_local_prompts or self.n_global_prompts > 1 or self.n_local_prompts > 1:
            template = self.template.replace("{}", " ").replace("_", " ").strip()
            tokenized_template = clip.tokenize(template)
            self.template_init_tokens = int(tokenized_template.argmax(dim=-1)) - 1
            self.n_token_context = self.template_init_tokens

            if self.learn_global_prompt or self.n_global_prompts > 1:
                if self.learn_global_prompt:
                    self.TRAINABLE_PARAMS.append("global_prompt")
                self.global_prompt = nn.Parameter(
                    torch.empty(self.n_global_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

            if self.learn_local_prompts or self.n_local_prompts > 1:
                if self.learn_local_prompts:
                    self.TRAINABLE_PARAMS.append("local_prompt")
                self.local_prompts = nn.Parameter(
                    torch.empty(self.n_local_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

        if self.n_neg_prompts > 0:
            self.TRAINABLE_PARAMS.append("neg_local_prompts")
            self.neg_local_prompts = nn.Parameter(
                torch.empty(self.n_neg_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
            )
            nn.init.normal_(self.neg_local_prompts, std=0.02)

        self.initialize_parameters()

        key_issue_clip = self.load_state_dict(clip_state_dict, strict=False)
        if len(key_issue_clip.missing_keys) > 0:
            lib.LOGGER.warning(f"Missing keys in CLIP: {key_issue_clip.missing_keys}")

        self.transformer = self.transformer if not self.parallel_text_encoder else vlp_tools.DataParallel(self.transformer)
        self.visual = self.visual if not self.parallel_vision_encoder else vlp_tools.DataParallel(self.visual)

    @property
    def num_devices(self) -> int:
        if not hasattr(self, "__device"):
            self.__device = torch.cuda.device_count()
        return self.__device

    def pad_if_necessary(self, x: Tensor) -> Tensor:
        if not self.parallel_text_encoder:
            return x, 0

        n = x.size(0)
        if n % self.num_devices == 0:
            return x, 0

        pad = self.num_devices - n % self.num_devices
        return torch.cat([x, torch.zeros(pad, *x.shape[1:], device=x.device)], dim=0), pad

    def unpad_if_necessary(self, x: Tensor, pad: int) -> Tensor:
        if pad == 0:
            return x

        return x[:-pad]

    def _default_encode_text(self, class_names: List[str]) -> Tensor:
        prompts = [self.template.format(name) for name in class_names]
        tokenized_text = clip.tokenize(prompts).cuda(non_blocking=True)
        text_features = super().encode_text(tokenized_text, batch_first=True)
        return text_features.unsqueeze(1)

    def _encode_text(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        x = torch.cat([prefix, prompt, suffix], dim=1)

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND  # This is not needed as we are using batch_first=True
        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # This is not needed as we are using batch_first=True
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_token_context] @ self.text_projection
        return x

    def _single_forward_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        n_prompts = prompts.size(0)
        n_classes = prefix.size(0)

        text_features = self._encode_text(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.repeat(n_classes, 1, 1),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        return text_features

    def _loop_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        text_features = []
        for i in range(prompts.size(0)):
            x = self._encode_text(prefix, prompts[i : i + 1].expand(prefix.size(0), -1, -1), suffix, eot_tokens)
            text_features.append(x)

        return torch.stack(text_features, dim=1)

    def _most_efficient_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        if self.parallel_text_encoder:
            return self._single_forward_encode_text(prefix, prompts, suffix, eot_tokens)
        return self._loop_encode_text(prefix, prompts, suffix, eot_tokens)

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        if not self.learn_global_prompt and not self.learn_local_prompts:
            text_features = self._default_encode_text(class_names)
            return text_features, text_features

        tokenized_text = clip.tokenize(class_names).cuda(non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        if self.learn_global_prompt or self.n_global_prompts > 1:
            global_prompt = self.global_prompt
            if self.prompts_batch_size < self.n_global_prompts and self.training:
                idx_select = torch.randperm(self.n_global_prompts)[: self.prompts_batch_size]  # we don't want to do this for local prompts
                global_prompt = self.global_prompt[idx_select]
            text_features = self._most_efficient_encode_text(prefix, global_prompt, suffix, eot_tokens)
        else:
            text_features = self._default_encode_text(class_names)

        if self.learn_local_prompts or self.n_local_prompts > 1:
            local_text_features = self._most_efficient_encode_text(prefix, self.local_prompts, suffix, eot_tokens)
        else:
            local_text_features = text_features

        return text_features, local_text_features

    def encode_image_and_proj(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        image_features, local_features = self.encode_image(image)

        if hasattr(self.visual, "proj"):
            image_features = image_features @ self.visual.proj
            if self.use_local_features:
                local_features = local_features @ self.visual.proj
        return image_features, local_features

    def forward(
        self,
        image: Tensor,
        class_names: Optional[List[str]] = None,
        text_features: Optional[Tensor] = None,
        local_text_features: Optional[Tensor] = None,
        pos_region_features: Optional[Tensor] = None,
        neg_region_features: Optional[Tensor] = None,
    ) -> Tensor:
        if class_names is not None:
            assert isinstance(class_names, list), "class_names must be a list of strings"
        if text_features is not None:
            assert isinstance(text_features, torch.Tensor), "text_features must be a Tensor"
        assert class_names is not None or text_features is not None, "Please provide either class_names or text_features"

        if text_features is None:
            assert local_text_features is None, "local_text_features should be None if text_features is None"
            text_features, local_text_features = self.encode_text(class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True) if self.learn_local_prompts else text_features

        image_features, local_features = self.encode_image_and_proj(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if pos_region_features is not None and neg_region_features is not None:
            local_text_features = self._prompt_features(self.local_prompts)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            pos_region_features = pos_region_features / pos_region_features.norm(dim=-1, keepdim=True)
            local_logits = torch.einsum("bkd,nd->bkn", pos_region_features, local_text_features)
            neg_local_logits = None
            if hasattr(self, "neg_local_prompts") and self.n_neg_prompts > 0:
                neg_local_text_features = self._prompt_features(self.neg_local_prompts)
                neg_local_text_features = neg_local_text_features / neg_local_text_features.norm(dim=-1, keepdim=True)
                neg_region_features = neg_region_features / neg_region_features.norm(dim=-1, keepdim=True)
                neg_local_logits = torch.einsum("bkd,nd->bkn", neg_region_features, neg_local_text_features)
            global_logits = None
            return global_logits, local_logits, neg_local_logits

        global_logits = torch.einsum("bd,kmd-> bkm", image_features, text_features)

        if self.use_local_features:
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            local_logits = torch.einsum("bpd,knd-> bpkn", local_features, local_text_features)
            neg_local_logits = None
            if hasattr(self, "neg_local_prompts") and self.n_neg_prompts > 0:
                neg_local_text_features = self.encode_neg_local_prompts(class_names)
                neg_local_logits = torch.einsum("bpd,knd-> bpkn", local_features, neg_local_text_features)
        else:
            local_logits = None
            neg_local_logits = None

        return global_logits, local_logits, neg_local_logits

    def _prompt_features(self, promtps: Tensor) -> Tensor:
        tokenized_text = clip.tokenize("").cuda(non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        text_features = self._most_efficient_encode_text(prefix, promtps, suffix, eot_tokens)
        return text_features

    def prompt_features(
        self,
    ) -> Tensor:
        global_prompt_features = local_prompt_features = None
        if self.learn_global_prompt:
            global_prompt_features = self._prompt_features(self.global_prompt)

        if self.learn_local_prompts:
            local_prompt_features = self._prompt_features(self.local_prompts)

        return global_prompt_features, local_prompt_features

    @property
    def device(self) -> torch.device:
        return self.text_projection.device

    def freeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = False

        for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            module.eval()
            module.train = lambda _: None

    def unfreeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = True

        for _ in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            print("Warning this module has Batchnorm that cannot be unfrozen.")
            break

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.state_dict().items() if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])}

    def load_trainable_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> _IncompatibleKeys:
        keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in keys.missing_keys if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])]
        if strict:
            error_msgs: List[str] = []
            if len(keys.unexpected_keys) > 0:
                error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in keys.unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))

            if len(error_msgs) > 0:
                raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=keys.unexpected_keys)

    @torch.no_grad()
    def initialize_prompt(self) -> NoneType:
        if not self.learn_global_prompt and not self.learn_local_prompts:
            return

        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        embedding = self.token_embedding(tokenized_template).type(self.dtype)
        global_prompt_init = embedding[:, 1 : 1 + self.template_init_tokens, :]

        if self.learn_global_prompt:
            self.global_prompt.data[:, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.n_global_prompts, -1, -1))

        if self.learn_local_prompts:
            self.local_prompts.data[:, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.n_local_prompts, -1, -1))

    def compute_gl_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        scores = -np.max(global_probs, axis=-1)

        if local_logits is not None:
            local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            local_score = -np.max(local_probs, axis=(1, 2))
            scores += local_score

        return scores

    def compute_L_mcm_scores(
        self,
        local_logits: Tensor,
    ) -> NoneType:
        assert local_logits is not None
        local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
        local_score = -np.max(local_probs, axis=(1, 2))
        return local_score

    def compute_mcm_scores(
        self,
        global_logits: Tensor,
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        global_score = -np.max(global_probs, axis=-1)
        return global_score

    def compute_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        ood_method: Optional[str] = None,
    ) -> NoneType:
        if ood_method is None:
            ood_method = self.ood_method

        if ood_method == "GL-MCM":
            return self.compute_gl_scores(global_logits, local_logits)
        elif ood_method == "MCM":
            return self.compute_mcm_scores(global_logits)
        elif ood_method == "L-MCM":
            return self.compute_L_mcm_scores(local_logits)
        else:
            raise ValueError(f"Method {self.ood_method} not implemented")

    @torch.no_grad()
    def create_prediction_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            local_logits = vlp_tools.topk_reduce(local_logits, topk=self.topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs

    def encode_neg_local_prompts(self, class_names: List[str]) -> torch.Tensor:
        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        eot_tokens = tokenized_template.argmax(dim=-1)
        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_template)
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]
        neg_text_features = self._most_efficient_encode_text(prefix, self.neg_local_prompts, suffix, eot_tokens)
        return neg_text_features

def select_regions_by_text_similarity(region_features, text_feature, k, sim_threshold=0.4):
    """
    输入：
        region_features: [N, D] 或 [B, N, D]
        text_feature: [D] 或 [B, D]
        k: int, 正/负样本数量
        sim_threshold: float, 相似度阈值
    输出：
        pos_indices, neg_indices: 正负样本索引
    """
    if region_features.dim() == 2:
        # 单张图片
        sim = F.cosine_similarity(region_features, text_feature.unsqueeze(0), dim=-1)  # [N]
        mask = sim > sim_threshold
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            pos_indices = sim.topk(k).indices  # 若无高于阈值的区域，退化为topk
        else:
            k_eff = min(k, valid_indices.numel())
            pos_indices = sim[valid_indices].topk(k_eff).indices
            pos_indices = valid_indices[pos_indices]
        # 负样本为低于阈值的区域，若不足k则补足
        neg_mask = ~mask
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if neg_indices.numel() < k:
            extra = sim.topk(k, largest=False).indices
            neg_indices = torch.cat([neg_indices, extra])[:k]
        else:
            neg_indices = neg_indices[:k]
        return pos_indices, neg_indices
    elif region_features.dim() == 3:
        # 批量图片
        B, N, D = region_features.shape
        pos_indices, neg_indices = [], []
        for b in range(B):
            sim = F.cosine_similarity(region_features[b], text_feature[b].unsqueeze(0), dim=-1)  # [N]
            mask = sim > sim_threshold
            valid_indices = mask.nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                pos_idx = sim.topk(k).indices
            else:
                k_eff = min(k, valid_indices.numel())
                pos_idx = sim[valid_indices].topk(k_eff).indices
                pos_idx = valid_indices[pos_idx]
            neg_mask = ~mask
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            if neg_idx.numel() < k:
                extra = sim.topk(k, largest=False).indices
                neg_idx = torch.cat([neg_idx, extra])[:k]
            else:
                neg_idx = neg_idx[:k]
            pos_indices.append(pos_idx)
            neg_indices.append(neg_idx)
        pos_indices = torch.stack(pos_indices)
        neg_indices = torch.stack(neg_indices)
        return pos_indices, neg_indices
    else:
        raise ValueError("region_features shape must be [N, D] or [B, N, D]")
