"""Unfused MultiheadAttention — splits fused QKV for HF PEFT compatibility.

``nn.MultiheadAttention`` uses a single ``in_proj_weight`` [3d, d] for Q, K, V.
HF PEFT needs separate ``nn.Linear`` modules (``q_proj``, ``k_proj``, ``v_proj``).

This module provides:

1. ``UnfusedMultiheadAttention``: drop-in replacement with separate projections.
2. ``unfuse_mha_layers``: replace ``self_attn`` in selected encoder layers.
3. ``refuse_mha_layers``: convert back to ``nn.MultiheadAttention`` after merge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnfusedMultiheadAttention(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` with separate Q/K/V.

    Splits ``in_proj_weight`` [3d, d] into three ``nn.Linear`` modules:

    - ``q_proj``: (d, d)
    - ``k_proj``: (d, d)
    - ``v_proj``: (d, d)

    HF PEFT can then target these by name
    (e.g. ``target_modules=["q_proj", "v_proj"]``).
    """

    def __init__(self, orig_mha: nn.MultiheadAttention):
        super().__init__()

        d = orig_mha.embed_dim
        nh = orig_mha.num_heads
        has_bias = orig_mha.in_proj_bias is not None

        # Separate Q / K / V projections
        self.q_proj = nn.Linear(d, d, bias=has_bias)
        self.k_proj = nn.Linear(d, d, bias=has_bias)
        self.v_proj = nn.Linear(d, d, bias=has_bias)

        # Copy weights from fused in_proj_weight [3d, d]
        with torch.no_grad():
            self.q_proj.weight.copy_(orig_mha.in_proj_weight[:d])
            self.k_proj.weight.copy_(orig_mha.in_proj_weight[d : 2 * d])
            self.v_proj.weight.copy_(orig_mha.in_proj_weight[2 * d :])
            if has_bias:
                self.q_proj.bias.copy_(orig_mha.in_proj_bias[:d])
                self.k_proj.bias.copy_(orig_mha.in_proj_bias[d : 2 * d])
                self.v_proj.bias.copy_(orig_mha.in_proj_bias[2 * d :])

        # Output projection (keep original)
        self.out_proj = orig_mha.out_proj

        self.embed_dim = d
        self.num_heads = nh
        self.head_dim = d // nh
        self.batch_first = getattr(orig_mha, "batch_first", True)
        self._qkv_same_embed_dim = True

        # Compatibility attributes for TransformerEncoderLayer fast-path checks.
        # Setting these to None forces the slow (standard) forward path.
        self.in_proj_weight = None
        self.in_proj_bias = None

    # ------------------------------------------------------------------ #
    #  Forward — matches nn.MultiheadAttention interface
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        # Handle batch_first=False: (S, B, D) → (B, S, D)
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, S, D = query.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Multi-head reshape: (B, S, D) → (B, nh, S, hd)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Build attention mask for SDPA
        sdpa_mask = None
        if key_padding_mask is not None:
            sdpa_mask = torch.zeros(
                B, 1, 1, K.size(2), dtype=Q.dtype, device=Q.device,
            )
            sdpa_mask.masked_fill_(
                key_padding_mask.bool().unsqueeze(1).unsqueeze(2), float("-inf"),
            )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                float_attn = torch.zeros_like(attn_mask, dtype=Q.dtype)
                float_attn.masked_fill_(attn_mask, float("-inf"))
                attn_mask = float_attn
            if sdpa_mask is None:
                sdpa_mask = attn_mask
            else:
                sdpa_mask = sdpa_mask + attn_mask

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=sdpa_mask, is_causal=is_causal,
        )

        # (B, nh, S, hd) → (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)

        # Restore (S, B, D) if batch_first=False
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None

    # ------------------------------------------------------------------ #
    #  Re-fuse: convert back to nn.MultiheadAttention
    # ------------------------------------------------------------------ #
    def to_mha(self) -> nn.MultiheadAttention:
        """Merge separate Q/K/V back into fused ``nn.MultiheadAttention``."""
        has_bias = self.q_proj.bias is not None
        d = self.embed_dim
        mha = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=self.num_heads,
            batch_first=self.batch_first,
            bias=has_bias,
            device=self.q_proj.weight.device,
        )
        with torch.no_grad():
            mha.in_proj_weight[:d].copy_(self.q_proj.weight)
            mha.in_proj_weight[d : 2 * d].copy_(self.k_proj.weight)
            mha.in_proj_weight[2 * d :].copy_(self.v_proj.weight)
            if has_bias:
                mha.in_proj_bias[:d].copy_(self.q_proj.bias)
                mha.in_proj_bias[d : 2 * d].copy_(self.k_proj.bias)
                mha.in_proj_bias[2 * d :].copy_(self.v_proj.bias)
            mha.out_proj.weight.copy_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                mha.out_proj.bias.copy_(self.out_proj.bias)
        return mha


# ------------------------------------------------------------------ #
#  Utilities
# ------------------------------------------------------------------ #

def unfuse_mha_layers(
    model: nn.Module,
    encoder_layers: list[nn.Module],
    layer_indices: list[int],
) -> None:
    """Replace ``self_attn`` (nn.MHA) with ``UnfusedMultiheadAttention``
    in the specified encoder layers."""
    for idx in layer_indices:
        layer = encoder_layers[idx]
        if isinstance(layer.self_attn, nn.MultiheadAttention):
            layer.self_attn = UnfusedMultiheadAttention(layer.self_attn)


def refuse_mha_layers(model: nn.Module) -> None:
    """Convert all ``UnfusedMultiheadAttention`` back to ``nn.MultiheadAttention``.

    No-op if no ``UnfusedMultiheadAttention`` modules are found.
    Safe to call on any model (including Geneformer which was never unfused).
    """
    replacements: list[tuple[str, str, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, UnfusedMultiheadAttention):
            mha = module.to_mha()
            if "." in name:
                parent_path, attr = name.rsplit(".", 1)
            else:
                parent_path, attr = "", name
            replacements.append((parent_path, attr, mha))

    for parent_path, attr, mha in replacements:
        parent = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        setattr(parent, attr, mha)
