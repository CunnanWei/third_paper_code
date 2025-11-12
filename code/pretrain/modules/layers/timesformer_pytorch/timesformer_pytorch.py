import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timesformer_pytorch.rotary import (
    apply_rot_emb,
    AxialRotaryEmbedding,
    RotaryEmbedding,
)

from timm.layers import DropPath


# -------------------------------------------------
# helpers
# -------------------------------------------------


def exists(val):
    return val is not None


# -------------------------------------------------
# layer wrappers
# -------------------------------------------------


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


# -------------------------------------------------
# token shift
# -------------------------------------------------


def shift(t: torch.Tensor, amt: int):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))


class PreTokenShift(nn.Module):

    def __init__(self, frames: int, fn: nn.Module):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, "b (f n) d -> b f n d", f=f)

        dim_chunk = dim // 3
        chunks = x.split(dim_chunk, dim=-1)
        shifted_chunks = tuple(shift(c, s) for c, s in zip(chunks[:3], (-1, 0, 1)))
        x = torch.cat((*shifted_chunks, *chunks[3:]), dim=-1)
        x = rearrange(x, "b f n d -> b (f n) d")
        x = torch.cat((cls_x, x), dim=1)
        return self.fn(x, *args, **kwargs)


# -------------------------------------------------
# feed‑forward
# -------------------------------------------------


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# -------------------------------------------------
# attention (SDPA)
# -------------------------------------------------


class Attention(nn.Module):
    def __init__(
        self, dim: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.0
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout_p = dropout
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        einops_from: str,
        einops_to: str,
        mask: torch.Tensor | None = None,
        cls_mask: torch.Tensor | None = None,
        rot_emb: torch.Tensor | None = None,
        **einops_dims,
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, :1], t[:, 1:]), (q, k, v)
        )

        cls_out = F.scaled_dot_product_attention(
            cls_q,
            k,
            v,
            attn_mask=cls_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        q_, k_, v_ = map(
            lambda t: rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims),
            (q_, k_, v_),
        )

        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        r = q_.shape[0] // cls_k.shape[0]
        cls_k = cls_k.repeat_interleave(r, dim=0)
        cls_v = cls_v.repeat_interleave(r, dim=0)

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)
        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# -------------------------------------------------
# Patch‑wise Lead Attention
# -------------------------------------------------


class PatchWiseLeadAttention(nn.Module):
    def __init__(
        self,
        dim_per_lead: int,
        num_leads: int = 12,
        heads: int = 4,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        assert dim_per_lead % heads == 0, "dim_per_lead 必须能被 heads 整除"
        self.num_leads = num_leads
        self.heads = heads

        self.to_qkv = nn.Linear(dim_per_lead, dim_per_lead * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim_per_lead, dim_per_lead), nn.Dropout(proj_dropout)
        )

    def forward(self, x: torch.Tensor):  # x: (b, n_patch, num_leads, dim_per_lead)
        b, n_patch, n_leads, dim = x.shape
        h = self.heads
        d_head = dim // h

        # qkv: (b, n_patch, num_leads, 3, h, d_head)
        qkv = (
            self.to_qkv(x)
            .reshape(b, n_patch, n_leads, 3, h, d_head)
            .permute(3, 0, 1, 4, 2, 5)
        )  # → (3, b, n_patch, h, num_leads, d_head)
        q, k, v = qkv

        # 合并 (b, n_patch, h) 为 batch 维，得到 (B', num_leads, d_head)
        B_ = b * n_patch * h
        q = q.reshape(B_, n_leads, d_head)
        k = k.reshape(B_, n_leads, d_head)
        v = v.reshape(B_, n_leads, d_head)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )  # (B', num_leads, d_head)

        # 还原形状
        out = (
            out.reshape(b, n_patch, h, n_leads, d_head)
            .transpose(2, 3)
            .reshape(b, n_patch, n_leads, dim)
        )
        return self.to_out(out)


# -------------------------------------------------
# Basic Residual Block
# -------------------------------------------------


class BasicBlock(nn.Module):
    """ResNet-style Basic Block for each lead"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# -------------------------------------------------
# Patch Embed with Lead Awareness
# -------------------------------------------------


class LeadGroupedPatchEmbed(nn.Module):
    """
    按导联分组的Patch Embedding，每个导联的3个模态共享卷积核：
    - Lead0: GADF0+RP0+MTF0 共享一个卷积核
    - Lead1: GADF1+RP1+MTF1 共享一个卷积核
    - ...
    - Lead11: GADF11+RP11+MTF11 共享一个卷积核
    """

    def __init__(
        self,
        *,
        in_ch: int = 36,
        patch_size: int = 16,
        dim: int = 192,
        num_leads: int = 12,
        num_modalities: int = 3,
        enable_lead_interaction: bool = True,
        lead_attn_heads: int = 4,
        pointwise_mix: bool = True,
    ):
        super().__init__()
        assert (
            in_ch == num_leads * num_modalities
        ), f"输入通道数应为 {num_leads * num_modalities}"
        assert dim % num_leads == 0, "dim 必须能被导联数整除"

        self.num_leads = num_leads
        self.num_modalities = num_modalities
        self.dim_per_lead = dim // num_leads
        self.enable_lead_interaction = enable_lead_interaction

        # 12组ResNet-style卷积，每个导联一组
        self.lead_convs = nn.ModuleList()
        for _ in range(num_leads):
            # 每个导联的ResNet结构
            lead_conv = nn.Sequential(
                # 初始下采样: 3 -> 64, stride=5 (100->20)
                nn.Conv2d(num_modalities, 64, kernel_size=5, stride=5, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # BasicBlock 1: 64 -> 64, stride=2 (20->10)
                BasicBlock(64, 64, stride=2),
                # BasicBlock 2: 64 -> 128, stride=1 (10->10)
                BasicBlock(64, 128, stride=1),
                # BasicBlock 3: 128 -> 128, stride=1 (10->10, 特征增强)
                BasicBlock(128, 128, stride=1),
                # 最终投影到目标维度
                nn.Conv2d(128, self.dim_per_lead, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dim_per_lead),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            self.lead_convs.append(lead_conv)

        self.mix = (
            nn.Conv2d(dim, dim, kernel_size=1, groups=1, bias=False)
            if pointwise_mix
            else nn.Identity()
        )

        # per‑lead LayerNorm
        self.lead_norm = nn.LayerNorm(self.dim_per_lead)

        if enable_lead_interaction:
            # 注意力机制在导联级别操作
            self.lead_attention = PatchWiseLeadAttention(
                dim_per_lead=self.dim_per_lead,
                num_leads=num_leads,  # 12个导联之间的交互
                heads=lead_attn_heads,
            )
        self.norm = nn.LayerNorm(dim)
        self.patch_size = patch_size


    def forward(self, x: torch.Tensor):  # x: (b, c, h, w)
        # 重新排列通道顺序（这里实际不需要重排，因为已经是导联分组）
        # x = self.reorder_channels(x)

        # 分别对每个导联进行三阶段卷积
        lead_features = []
        for i, conv_stages in enumerate(self.lead_convs):
            start_ch = i * self.num_modalities
            end_ch = (i + 1) * self.num_modalities
            lead_x = x[:, start_ch:end_ch, :, :]  # (b, num_modalities, h, w)
            lead_feat = conv_stages(lead_x)  # (b, dim_per_lead, h/p, w/p)
            lead_features.append(lead_feat)

        # 拼接所有导联的特征
        x = torch.cat(lead_features, dim=1)  # (b, dim, h/p, w/p)
        x = self.mix(x)
        x = x.flatten(2).transpose(1, 2)  # (b, n_patch, dim)

        # reshape -> (b, n_patch, num_leads, dim_per_lead)
        b, n_patch, dim = x.shape
        x = x.view(b, n_patch, self.num_leads, self.dim_per_lead)
        x = self.lead_norm(x)  # per‑lead LN

        if self.enable_lead_interaction:
            # 导联间注意力交互
            x = self.lead_attention(x)

        x = x.view(b, n_patch, dim)
        return self.norm(x)





# -------------------------------------------------
# TimeSformer
# -------------------------------------------------


class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 192,
        num_frames: int = 8,
        num_classes: int = 5,
        image_size: int = 100,
        patch_size: int = 10,
        channels: int = 36,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 24,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        rotary_emb: bool = True,
        shift_tokens: bool = False,
        groups: int = 12,
        enable_lead_interaction: bool = True,
        lead_attn_heads: int = 4,
        pointwise_mix: bool = True,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0

        self.num_frames = num_frames
        hp = wp = image_size // patch_size
        self.n_patches = hp * wp

        # patch embed
        self.patch_embed = LeadGroupedPatchEmbed(
            in_ch=channels,
            patch_size=patch_size,
            dim=dim,
            num_leads=12,  # 固定为12导联
            num_modalities=3,  # 固定为3个模态(GADF, RP, MTF)
            enable_lead_interaction=enable_lead_interaction,
            lead_attn_heads=lead_attn_heads,
            pointwise_mix=pointwise_mix,
        )

        self.cls_token = nn.Parameter(torch.randn(1, dim))

        # pos emb
        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(self.n_patches * num_frames + 1, dim)

        # Transformer layers
        self.layers = nn.ModuleList([])
        # 创建线性增长的drop_path_rate列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        for i in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            t_attn = Attention(
                dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
            )
            s_attn = Attention(
                dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
            )

            # DropPath layers
            drop_path = DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()

            if shift_tokens:
                t_attn, s_attn, ff = map(
                    lambda fn: PreTokenShift(num_frames, fn), (t_attn, s_attn, ff)
                )
            t_attn, s_attn, ff = map(lambda fn: PreNorm(dim, fn), (t_attn, s_attn, ff))

            self.layers.append(nn.ModuleList([t_attn, s_attn, ff, drop_path]))

        # 只使用 CLS token
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.hp = hp
        self.wp = wp

    def forward(self, video: torch.Tensor, mask: torch.Tensor | None = None):
        """video: (b, f, c, h, w)"""
        b, f, _, h, w = video.shape
        p = self.patch_embed.patch_size
        assert h % p == w % p == 0 and f == self.num_frames

        video = rearrange(video, "b f c h w -> (b f) c h w")
        tokens = self.patch_embed(video)
        tokens = rearrange(tokens, "(b f) n d -> b (f n) d", b=b, f=f)

        cls = repeat(self.cls_token, "1 d -> b 1 d", b=b)
        x = torch.cat((cls, tokens), dim=1)

        if not self.use_rotary_emb:
            pos = self.pos_emb(torch.arange(x.shape[1], device=x.device))
            x = x + pos
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=x.device)
            image_pos_emb = self.image_rot_emb(self.hp, self.wp, device=x.device)

        frame_mask = cls_mask = None  # 可按需生成 mask

        for t_attn, s_attn, ff, drop_path in self.layers:

            # 时间注意力
            x = x + drop_path(
                t_attn(
                    x,
                    "b (f n) d",
                    "(b n) f d",
                    n=self.n_patches,
                    mask=frame_mask,
                    cls_mask=cls_mask,
                    rot_emb=frame_pos_emb,
                )
            )
            # 空间注意力
            x = x + drop_path(
                s_attn(
                    x,
                    "b (f n) d",
                    "(b f) n d",
                    f=f,
                    cls_mask=cls_mask,
                    rot_emb=image_pos_emb,
                )
            )

            # 前馈网络
            x = x + drop_path(ff(x))

        # 只使用 CLS token
        return self.to_out(x[:, 0])
        # return x[:, 0, :] # (b, dim)