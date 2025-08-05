from typing import Type, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

from gallop.vlprompt.tools.topk_reduce import topk_reduce

NoneType = Type[None]


class GlobalLocalLoss(_WeightedLoss):

    def __init__(
        self,
        use_global_loss: bool = True,
        use_local_loss: bool = True,
        use_neg_local_loss: bool = False,
        topk: List[int] = [5],
        global_dropout_p: float = 0.75,
        neg_local_loss_weight: float = 1.0,
    ) -> NoneType:
        super().__init__()

        self.use_global_loss = use_global_loss
        self.use_local_loss = use_local_loss
        self.use_neg_local_loss = use_neg_local_loss
        self.topk = topk
        self.global_dropout_p = global_dropout_p
        self.neg_local_loss_weight = neg_local_loss_weight

    def mask_prompt_tokens(self, prompt_emb: Tensor, mask_p: float) -> Tensor:
        """
        对prompt embedding的token维度随机mask部分token（置零），prompt_emb: (num_class, prompt_len, embed_dim)
        """
        if mask_p <= 0 or prompt_emb.size(1) == 1:
            return prompt_emb
        device = prompt_emb.device
        num_tokens = prompt_emb.size(1)
        mask = (torch.rand(prompt_emb.size(0), num_tokens, device=device) > mask_p).float().unsqueeze(-1)
        return prompt_emb * mask

    def forward(
        self,
        global_logits: Tensor,
        local_logits: Tensor,
        targets: Tensor,
        logit_scale: float,
        neg_local_logits: Tensor = None,
        image_features: Tensor = None,
        text_features: Tensor = None,
    ) -> Tensor:
        """
        global_logits is a Tensor of shape (b, k, 1) or (b, k, n)
        local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
        neg_local_logits is a Tensor of shape (b, p, k, n_neg) or None
        """
        global_loss = local_loss = neg_local_loss = 0.

        if self.use_local_loss and local_logits is not None:
            local_logits = topk_reduce(local_logits, self.topk)
            local_loss = F.cross_entropy(logit_scale * local_logits, targets.unsqueeze(-1).expand(-1, local_logits.size(-1)))

        if self.use_neg_local_loss and neg_local_logits is not None:
            neg_local_logits = topk_reduce(neg_local_logits, self.topk)
            neg_local_loss = F.cross_entropy(logit_scale * neg_local_logits, targets.unsqueeze(-1).expand(-1, neg_local_logits.size(-1)))
            neg_local_loss = self.neg_local_loss_weight * neg_local_loss

        if self.use_global_loss:
            # Prompt Masking替换Dropout:
            if (image_features is not None) and (text_features is not None):
                # text_features: (num_class, prompt_len, embed_dim)
                masked_text_features = self.mask_prompt_tokens(text_features, self.global_dropout_p)
                # 池化未被mask的token
                pooled_text_features = masked_text_features.sum(dim=1) / (masked_text_features.abs().sum(dim=-1) > 0).sum(dim=1).clamp(min=1).unsqueeze(-1)
                # image_features: (b, embed_dim)
                global_logits = torch.einsum("bd,kd->bk", image_features, pooled_text_features)
                global_loss = F.cross_entropy(logit_scale * global_logits, targets)
            else:
                # 兼容旧接口，直接用原有logits
                keep_number = max(global_logits.size(-1) - int(self.global_dropout_p * global_logits.size(-1)), 1)
                index = torch.randint(global_logits.size(-1), (global_logits.size(0), 1, keep_number), device=global_logits.device).expand(-1, global_logits.size(1), -1)
                global_logits = global_logits.gather(-1, index).mean(-1)
                if global_logits.ndim == 2:
                    global_loss = F.cross_entropy(logit_scale * global_logits, targets)
                elif global_logits.ndim == 3:
                    global_loss = F.cross_entropy(logit_scale * global_logits, targets.unsqueeze(-1).expand(-1, global_logits.size(-1)))
                else:
                    raise ValueError(f"Global logits must have 2 or 3 dimensions, but got {global_logits.ndim}.")

        return global_loss + local_loss + neg_local_loss
