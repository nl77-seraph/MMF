"""MMF inference model that bypasses the support branch and feature-reweighting module.

The key observation in MMF is that the feature-reweighting module is a 1x1
depthwise dynamic convolution whose weights are the class-conditioned vectors
``W_c`` produced by the meta_learnet. When ``W_c`` is cached for each class, the
reweighting becomes a pure channel-wise broadcast multiply and the entire
support branch can be dropped at inference. This lets us:

- Ship a lean checkpoint without ``meta_learnet`` or ``feature_reweighting``.
- Scale to large monitored sets with zero support-branch compute per query.

This file implements ``CachedInferenceMMF`` which accepts ``query_data`` and a
pre-computed ``class_bank`` tensor of shape ``(N, C)`` and produces the same
``logits`` as the full pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models")))

from models.feature_extractors import DFFeatureExtractor  # noqa: E402
from models.classification_head_enhanced import EnhancedClassificationHead  # noqa: E402


class CachedInferenceMMF(nn.Module):
    """Inference-only MMF: backbone + classification head + cached W_c bank.

    Attributes
    ----------
    feature_extractor : DFFeatureExtractor
        The 1D-CNN backbone.
    classification_head : EnhancedClassificationHead
        TopM self-attn + CrossClass attn + MLP. ``num_classes`` can be any
        value ``<= class_bank.size(0)``; at inference we slice the bank to
        match the required ``N`` and re-create the head's pos-embed on the fly
        only if ``seq_len`` differs (not normally the case).
    """

    def __init__(
        self,
        feature_extractor: DFFeatureExtractor,
        classification_head: EnhancedClassificationHead,
        num_classes: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward(
        self,
        query_data: torch.Tensor,
        class_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Run cached-bank MMF inference.

        Parameters
        ----------
        query_data : (B, L_q) float tensor.
        class_bank : (N, C) float tensor with the cached reweighting vectors.

        Returns
        -------
        logits : (B, N) float tensor.
        """
        assert query_data.dim() == 2, f"query expected (B, L), got {query_data.shape}"
        assert class_bank.dim() == 2, f"class_bank expected (N, C), got {class_bank.shape}"
        B = query_data.size(0)
        N, C = class_bank.shape
        assert C == self.feature_dim, f"bank channel dim {C} != model feature_dim {self.feature_dim}"
        # The classification head has a pos_embed of shape (1, L', C) and an
        # output MLP that is shared across classes; so we can still use it
        # with an arbitrary ``N`` (only the internal ``self.num_classes`` is
        # used to split the batch dim).
        self.classification_head.num_classes = N

        # Backbone: (B, L_q) -> (B, L', C)
        query_features = self.feature_extractor.forward_full(query_data)
        # (B, L', C) -> (B, 1, L', C) broadcast-multiplied by (1, N, 1, C) class-wise gate.
        B_, L_p, C_ = query_features.shape
        assert B_ == B and C_ == C
        # Equivalent to the dynamic 1x1 depthwise conv in FeatureReweightingModule:
        #     out[b, n, c, t] = query[b, t, c] * class_bank[n, c]
        # Reshape to (B*N, C, L') to match the head's expected input.
        gate = class_bank.view(1, N, 1, C)                 # (1, N, 1, C)
        qf = query_features.unsqueeze(1)                   # (B, 1, L', C)
        reweighted = qf * gate                             # (B, N, L', C)
        reweighted = reweighted.permute(0, 1, 3, 2).contiguous()  # (B, N, C, L')
        reweighted = reweighted.view(B * N, C, L_p)        # (B*N, C, L')

        logits = self.classification_head(reweighted)       # (B, N)
        return logits


def build_cached_inference_model_from_full(
    full_model,
    num_classes: int,
) -> CachedInferenceMMF:
    """Extract ``feature_extractor`` and ``classification_head`` from a full net."""
    return CachedInferenceMMF(
        feature_extractor=full_model.feature_extractor,
        classification_head=full_model.classification_head,
        num_classes=num_classes,
        feature_dim=full_model.query_feature_dim,
    )


def build_lean_state_dict(full_model) -> dict:
    """Return a state dict containing only backbone + classification_head."""
    sd = {}
    for k, v in full_model.state_dict().items():
        if k.startswith("feature_extractor.") or k.startswith("classification_head."):
            sd[k] = v.detach().clone()
    return sd


def load_lean_checkpoint(
    feature_extractor: DFFeatureExtractor,
    classification_head: EnhancedClassificationHead,
    lean_state_dict: dict,
) -> None:
    """Load a ``build_lean_state_dict`` output into the two submodules in-place."""
    fe_sd = {k[len("feature_extractor."):]: v for k, v in lean_state_dict.items()
             if k.startswith("feature_extractor.")}
    ch_sd = {k[len("classification_head."):]: v for k, v in lean_state_dict.items()
             if k.startswith("classification_head.")}
    feature_extractor.load_state_dict(fe_sd, strict=True)
    # pos_embed might have a different seq_len if you change L_query; allow non-strict.
    classification_head.load_state_dict(ch_sd, strict=False)


@torch.no_grad()
def compute_class_bank(
    full_model,
    support_data: torch.Tensor,
    support_masks: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Run ``meta_learnet`` on ``support_data`` to produce ``(N, C)`` bank.

    Parameters
    ----------
    full_model : EnhancedMultiMetaFingerNet
        A trained MMF model (must include ``meta_learnet``).
    support_data : (N, K, L_s) float tensor.
    support_masks : (N, K, L_s) float tensor, optional. Defaults to all ones.
    chunk_size : int, optional. If provided, process classes in chunks to bound
        the peak memory.
    """
    assert support_data.dim() == 3, f"support expected (N, K, L), got {support_data.shape}"
    N = support_data.size(0)
    if support_masks is None:
        support_masks = torch.ones_like(support_data)
    full_model.eval()

    if chunk_size is None or chunk_size >= N:
        return full_model.support_forward(support_data, support_masks)

    out = []
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk = full_model.support_forward(support_data[s:e], support_masks[s:e])
        out.append(chunk)
    return torch.cat(out, dim=0)
