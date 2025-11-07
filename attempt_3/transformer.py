# transformer.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Attention backend dispatch
# =========================

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")
try:
    import xformers.ops as xops  # optional

    _HAS_XFORMERS = True
except Exception:
    _HAS_XFORMERS = False


def _configure_sdpa_kernels() -> None:
    """
    Prefer flash / memory efficient attention kernels when available on CUDA.
    Safe to call on CPU-only builds.
    """
    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        )
    except (AttributeError, RuntimeError):
        pass


def _resolve_is_causal(is_causal: Optional[bool]) -> bool:
    return bool(is_causal)


def attention_dispatch(q, k, v, attn_mask=None, is_causal: bool = False):
    """
    q, k, v: [B, H, T, Dh]
    attn_mask: None, [T, T], or broadcastable bool/float mask
    returns: [B, H, T, Dh]
    Priority: xFormers -> PyTorch SDPA -> manual
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if _HAS_XFORMERS and attn_mask is None:
        return xops.memory_efficient_attention(q, k, v, attn_bias=None, p=0.0)

    if _HAS_SDPA:
        B, H, T_q, Dh = q.shape
        T_k = k.shape[2]
        T_v = v.shape[2]
        if T_k != T_v:
            raise ValueError("Key and value sequence lengths must match.")
        q_ = q.reshape(B * H, T_q, Dh)
        k_ = k.reshape(B * H, T_k, Dh)
        v_ = v.reshape(B * H, T_v, Dh)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask_ = attn_mask[None].expand(B * H, -1, -1)
            elif attn_mask.dim() == 3:
                attn_mask_ = attn_mask
            else:
                raise ValueError("Unsupported attn_mask shape for SDPA")
        else:
            attn_mask_ = None
        out = F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=attn_mask_, is_causal=is_causal
        )
        return out.reshape(B, H, T_q, Dh)

    B, H, T_q, Dh = q.shape
    T_k = k.shape[2]
    scale = 1.0 / math.sqrt(Dh)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_q, T_k]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    if is_causal:
        if T_q != T_k:
            raise ValueError("Causal masking requires matching query/key lengths.")
        causal = torch.ones(T_q, T_k, device=q.device, dtype=torch.bool).triu(1)
        attn = attn.masked_fill(causal, float("-inf"))
    attn = attn.softmax(-1)
    return torch.matmul(attn, v)


_configure_sdpa_kernels()


# ===========
# Core blocks
# ===========


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, 2 * d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        u, v = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(u) * v)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

    def _build(self, device, dtype, T):
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}.")
        pos = torch.arange(T, device=device, dtype=dtype)
        freqs = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim)
        )
        theta = torch.einsum("t,f->tf", pos, freqs)  # [T, dim/2]
        cos = torch.cat([theta.cos(), theta.cos()], dim=-1)  # [T, dim]
        sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
        return cos[None, None, :, :], sin[None, None, :, :]

    def forward(self, T: int, device, dtype):
        return self._build(device, dtype, T)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        attn_dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.dh) if use_rope else None

    def _split_heads(self, x, B, T):
        return x.view(B, T, self.heads, self.dh).transpose(1, 2)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q, B, T)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        if self.use_rope:
            cos, sin = self.rope(T, x.device, x.dtype)
            q, k = apply_rope(q, k, cos, sin)

        out = attention_dispatch(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(self.dropout(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_rope=False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, attn_dropout, use_rope=use_rope)
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            SwiGLU(dim, int(dim * mlp_ratio)),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ff(self.norm2(x))
        return x


# ============================
# Modality-time embeddings
# ============================


class ModalityTimeEmbedding(nn.Module):
    """
    Projects scalar time-series modalities into Transformer token space.
    Expected input shape: [B, T, F, value_dim] where value_dim defaults to 1.
    """

    def __init__(
        self,
        num_modalities: int,
        max_time: int,
        d_model: int,
        value_dim: int = 1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.max_time = max_time
        self.d_model = d_model
        self.value_dim = value_dim

        self.value_proj = nn.Linear(value_dim, d_model, bias=False)
        self.time_embedding = nn.Embedding(max_time, d_model)
        self.modality_embedding = nn.Embedding(num_modalities, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected input shape [B, T, F, value_dim].")
        B, T, F, D = x.shape
        if D != self.value_dim:
            raise ValueError(f"Expected value_dim {self.value_dim}, got {D}.")
        if F != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {F}.")
        if T > self.max_time:
            raise ValueError(f"Sequence length {T} exceeds configured max_time {self.max_time}.")

        values = self.value_proj(x)  # [B, T, F, d_model]
        time_emb = self.time_embedding(torch.arange(T, device=x.device))  # [T, d_model]
        mod_emb = self.modality_embedding(torch.arange(F, device=x.device))  # [F, d_model]

        tokens = values + time_emb.unsqueeze(0).unsqueeze(2) + mod_emb.unsqueeze(0).unsqueeze(1)
        return tokens.view(B, T * F, self.d_model)


# ============================
# General-purpose Transformer
# ============================


class SwiGLUProjection(nn.Module):
    """
    Two-layer projection with SwiGLU non-linearity between linear blocks.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return self.fc2(x)


class SelfAttentionTransformer(nn.Module):
    """
    Self-attention encoder that can ingest raw scalar modalities or pre-computed embeddings.

    When `num_modalities` and `max_time` are provided, the model automatically applies
    `ModalityTimeEmbedding` to inputs shaped [B, T, F, value_dim]. Otherwise it expects
    dense embeddings of shape [B, S, d_model] (or [B, S, input_dim] if an input projection
    dimension is supplied).
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_modalities: Optional[int] = None,
        max_time: Optional[int] = None,
        value_dim: int = 1,
        input_dim: Optional[int] = None,
        use_rope: bool = False,
    ):
        super().__init__()
        if (num_modalities is None) ^ (max_time is None):
            raise ValueError("num_modalities and max_time must be specified together.")

        self.embedder = (
            ModalityTimeEmbedding(num_modalities, max_time, d_model, value_dim=value_dim)
            if num_modalities is not None
            else None
        )

        self.input_projection = None
        self.output_projection = None
        if self.embedder is None and input_dim is not None and input_dim != d_model:
            self.input_projection = SwiGLUProjection(input_dim, d_model, d_model)
            self.output_projection = SwiGLUProjection(d_model, d_model, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    heads,
                    mlp_ratio,
                    attn_dropout,
                    ff_dropout=dropout,
                    use_rope=use_rope,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.embedder is not None:
            tokens = self.embedder(x)
        else:
            if x.dim() != 3:
                raise ValueError("Expected embedded input shape [B, S, D].")
            tokens = x
            if self.input_projection is not None:
                tokens = self.input_projection(tokens)

        tokens = self.dropout(tokens)
        tokens = tokens.contiguous()

        causal = _resolve_is_causal(is_causal)
        for layer in self.layers:
            tokens = layer(tokens, attn_mask=attn_mask, is_causal=causal)

        tokens = self.norm(tokens)
        if self.output_projection is not None:
            tokens = self.output_projection(tokens)
        return tokens



# ==========================================
# Cross-attention token refiner (single token)
# ==========================================

class CrossAttention(nn.Module):
    """
    Single-query multi-head cross-attention.
    Q comes from a 1-token query, K/V from a context sequence.
    Shapes:
      token:   [B, 1, D]
      context: [B, S, D]
      return:  [B, 1, D]
    """
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(attn_dropout)

    def _split_heads(self, x: torch.Tensor, B: int, T: int):
        # [B, T, D] -> [B, H, T, Dh]
        return x.view(B, T, self.heads, self.dh).transpose(1, 2)

    def forward(self, token: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, Tq, D = token.shape          # Tq == 1
        _, Tk, _ = context.shape

        q = self._split_heads(self.q_proj(token), B, Tq)     # [B,H,1,Dh]
        k = self._split_heads(self.k_proj(context), B, Tk)   # [B,H,S,Dh]
        v = self._split_heads(self.v_proj(context), B, Tk)   # [B,H,S,Dh]

        # attention_dispatch expects [B,H,T,D]; our query len is 1
        out = attention_dispatch(q, k, v, attn_mask=attn_mask, is_causal=False)  # [B,H,1,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)                    # [B,1,D]
        return self.o_proj(self.drop(out))


class CrossAttnBlock(nn.Module):
    """
    Pre-norm cross-attn + SwiGLU MLP on the single token.
    Context is normalized but otherwise passed through unchanged.
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, attn_dropout: float = 0.0, ff_dropout: float = 0.0):
        super().__init__()
        self.norm_q = RMSNorm(dim)
        self.norm_ctx = RMSNorm(dim)
        self.cross = CrossAttention(dim, heads, attn_dropout)
        self.norm_ff = RMSNorm(dim)
        self.ff = nn.Sequential(
            SwiGLU(dim, int(dim * mlp_ratio)),
            nn.Dropout(ff_dropout),
        )

    def forward(self, token: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Cross-attend (token queries → context keys/values)
        token = token + self.cross(self.norm_q(token), self.norm_ctx(context), attn_mask=attn_mask)
        # Token-only MLP
        token = token + self.ff(self.norm_ff(token))
        return token


class CrossAttentionTransformer(nn.Module):
    """
    Generates a new learnable token (size d_model) and refines it via iterative
    cross-attention over time-series tokens or pre-embedded inputs.

    Input modes:
      - Raw time-series: x shape [B, T, F, value_dim] with (num_modalities, max_time)
      - Embedded:        x shape [B, S, input_dim] (set input_dim; no embedder used)

    Output:
      - token: [B, d_model]    (the enriched single token)
      - (optional) you can expose attention maps if you want; kept minimal here.
    """
    def __init__(
        self,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_modalities: Optional[int] = None,
        max_time: Optional[int] = None,
        value_dim: int = 1,
        input_dim: Optional[int] = None,
        token_init: str = "learned",   # "learned" | "mean" | "zero"
    ):
        super().__init__()
        if (num_modalities is None) ^ (max_time is None):
            raise ValueError("num_modalities and max_time must be specified together.")

        # Context embedding path (reuse your embedder/projection logic)
        self.embedder = (
            ModalityTimeEmbedding(num_modalities, max_time, d_model, value_dim=value_dim)
            if num_modalities is not None
            else None
        )
        self.input_projection = None
        if self.embedder is None and input_dim is not None and input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)

        self.context_dropout = nn.Dropout(dropout)

        # Single query token initialization
        self.token_init = token_init
        if token_init == "learned":
            self.learned_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.learned_token, std=0.02)
        else:
            self.learned_token = None  # for "mean" or "zero" init

        # Depth of cross-attention refinement
        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, heads, mlp_ratio, attn_dropout, ff_dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(d_model)

    def _embed_context(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedder is not None:
            ctx = self.embedder(x)               # [B, T*F, D]
        else:
            if x.dim() != 3:
                raise ValueError("Expected embedded input shape [B, S, D].")
            ctx = x
            if self.input_projection is not None:
                ctx = self.input_projection(ctx)  # [B, S, D]
        return self.context_dropout(ctx.contiguous())

    def _init_token(self, ctx: torch.Tensor) -> torch.Tensor:
        B, S, D = ctx.shape
        if self.token_init == "learned":
            return self.learned_token.expand(B, -1, -1)                 # [B,1,D]
        elif self.token_init == "mean":
            return ctx.mean(dim=1, keepdim=True)                        # [B,1,D]
        elif self.token_init == "zero":
            return ctx.new_zeros(B, 1, D)                               # [B,1,D]
        else:
            raise ValueError(f"Unknown token_init '{self.token_init}'")

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: raw time-series [B,T,F,value_dim]  OR embedded [B,S,D]
        attn_mask (optional): broadcastable mask over context positions, or None
        returns: enriched token [B, D]
        """
        context = self._embed_context(x)           # [B,S,D]
        token = self._init_token(context)          # [B,1,D]

        for layer in self.layers:
            token = layer(token, context, attn_mask=attn_mask)

        token = self.final_norm(token)             # [B,1,D]
        return token.squeeze(1)                    # [B,D]
