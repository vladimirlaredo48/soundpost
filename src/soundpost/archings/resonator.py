"""Three-dimensional encoders for the five-phase volume.

Ref: Eq. (3), F = E_theta(V). The default is a selective state-space (Mamba)
encoder with a depthwise local branch over the patch grid; the swin and resnet
variants exist for the backbone-swap ablation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from soundpost.settings import EncoderConfig, EncoderKind, VolumeSpec


class SelectiveScan(nn.Module):
    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1))
        )
        self.d_skip = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        proj = self.x_proj(x)
        dt, b_mat, c_mat = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a = -torch.exp(self.a_log)
        delta_a = torch.exp(torch.einsum("bln,nm->blnm", dt, a))
        delta_bx = torch.einsum("bln,blm,bln->blnm", dt, b_mat, x)
        state = x.new_zeros(b, self.d_inner, self.d_state)
        outputs = []
        for t in range(length):
            state = delta_a[:, t] * state + delta_bx[:, t]
            outputs.append(torch.einsum("bnm,bm->bn", state, c_mat[:, t]))
        y = torch.stack(outputs, dim=1)
        return y + x * self.d_skip


class MambaBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d_inner = cfg.d_model * cfg.expand
        self.bidirectional = cfg.bidirectional
        self.norm = nn.LayerNorm(cfg.d_model)
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, cfg.conv_kernel, groups=d_inner, padding=cfg.conv_kernel - 1
        )
        self.scan = SelectiveScan(d_inner, cfg.d_state)
        self.scan_rev = SelectiveScan(d_inner, cfg.d_state) if cfg.bidirectional else None
        self.local = nn.Conv3d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _branch(self, x: Tensor, grid: tuple[int, int, int]) -> Tensor:
        length = x.shape[1]
        conv = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        conv = F.silu(conv)
        fwd = self.scan(conv)
        if self.scan_rev is not None:
            rev = torch.flip(self.scan_rev(torch.flip(conv, dims=[1])), dims=[1])
            fwd = fwd + rev
        local = rearrange(conv, "b (d h w) c -> b c d h w", d=grid[0], h=grid[1], w=grid[2])
        local = rearrange(self.local(local), "b c d h w -> b (d h w) c")
        return fwd + local

    def forward(self, x: Tensor, grid: tuple[int, int, int]) -> Tensor:
        residual = x
        x = self.norm(x)
        gated, skip = self.in_proj(x).chunk(2, dim=-1)
        y = self._branch(gated, grid) * F.silu(skip)
        return residual + self.drop(self.out_proj(y))


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, v: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        feat = self.proj(v)
        grid = (feat.shape[2], feat.shape[3], feat.shape[4])
        return rearrange(feat, "b c d h w -> b (d h w) c"), grid


class StateSpaceEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, vol: VolumeSpec) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        in_ch = vol.n_phases * vol.channels
        self.embed = _PatchEmbed(in_ch, cfg.d_model, cfg.patch_size)
        self.blocks = nn.ModuleList(MambaBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, v: Tensor) -> Tensor:
        tokens, grid = self.embed(v)
        for block in self.blocks:
            tokens = block(tokens, grid)
        return self.norm(tokens)


class _ResBlock3d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.body(x))


class ResNet3dEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, vol: VolumeSpec) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        in_ch = vol.n_phases * vol.channels
        self.stem = nn.Conv3d(in_ch, cfg.d_model, cfg.patch_size, stride=cfg.patch_size)
        self.blocks = nn.Sequential(*[_ResBlock3d(cfg.d_model) for _ in range(cfg.depth // 3)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, v: Tensor) -> Tensor:
        feat = self.blocks(self.stem(v))
        tokens = rearrange(feat, "b c d h w -> b (d h w) c")
        return self.norm(tokens)


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (three h d) -> three b h n d", three=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.proj(out)


class _SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class SwinLikeEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, vol: VolumeSpec) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        in_ch = vol.n_phases * vol.channels
        self.embed = _PatchEmbed(in_ch, cfg.d_model, cfg.patch_size)
        heads = max(1, cfg.d_model // 64)
        self.layers = nn.ModuleList(_SwinBlock(cfg.d_model, heads) for _ in range(cfg.depth // 2))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, v: Tensor) -> Tensor:
        tokens, _ = self.embed(v)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


def build_encoder(cfg: EncoderConfig, vol: VolumeSpec) -> nn.Module:
    if cfg.kind is EncoderKind.ssm:
        return StateSpaceEncoder(cfg, vol)
    if cfg.kind is EncoderKind.resnet3d:
        return ResNet3dEncoder(cfg, vol)
    if cfg.kind is EncoderKind.swin_unetr:
        return SwinLikeEncoder(cfg, vol)
    raise ValueError(f"unknown encoder kind {cfg.kind}")


def positional_grid(size: tuple[int, int, int], patch: int) -> int:
    return math.prod(s // patch for s in size)
