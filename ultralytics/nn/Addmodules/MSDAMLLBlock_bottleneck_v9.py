# 新模块文件: ultralytics/nn/Addmodules/MSDAMLLBlock.py
# 该文件定义了YOLO11的修改版C3k模块，用于道路交通检测任务（数据集TJU-DHD-TRAFFIC），其中C3中的瓶颈模块被替换为MSDAMLLABlock（MultiDilatelocalAttention后跟MLLABlock的串联）。
# 已修复TypeError in RoPE: channel_dims转换为list of int，支持矩形特征图的2D RoPE。
# 确保代码符合YOLO11结构：继承自ultralytics.nn.modules.block.C3，兼容B,C,H,W输入/输出，并可直接在YAML中替换。

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from ultralytics.nn.modules import C3, Conv  # Ultralytics标准导入
from ultralytics.nn.modules.conv import RepConv

__all__ = ['MSDAMLLABlock', 'ELAN1', 'RepNCSPELAN4', 'RepNCSPELAN4_MSDAMLLA_bottleneck']

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RoPE(nn.Module):
    r"""Rotary Positional Embedding."""
    def __init__(self, base=10000):
        super().__init__()
        self.base = base

    def generate_rotations(self, x):
        channel_dims = list(x.shape[1:-1])  # 转换为list of int，确保兼容矩形
        feature_dim = x.shape[-1]
        num_dims = len(channel_dims)
        if num_dims == 0:
            return torch.tensor(1.0, device=x.device)  # 边界处理
        k_max = feature_dim // (2 * num_dims)
        assert feature_dim % (2 * num_dims) == 0, "Feature dimension must be divisible by 2 * number of channel dimensions"

        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        grid = torch.meshgrid([torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims], indexing='ij')
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in grid], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(-1)
        rotations_im = torch.sin(angles).unsqueeze(-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    def forward(self, x):
        rotations = self.generate_rotations(x)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations) * x_complex
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
    r"""Linear Attention with LePE and RoPE."""

    def __init__(self, dim, num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x  # b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 应用2D RoPE
        q_rope = self.rope(q.reshape(b, H, W, c)).reshape(b, n, c).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, H, W, c)).reshape(b, n, c).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v_2d = v.transpose(1, 2).reshape(b, H, W, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v_2d).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

class MLLABlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads  # 固定传入num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H * W * C == x.numel() / B, "Input shape mismatch"
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, H * W, C)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).reshape(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = self.attn(x, H, W)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

# MSDA多尺度空洞卷积 (保持不变)
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilatelocalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads {num_heads} must be multiple of num_dilation {self.num_dilation}"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList([DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i]) for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"Channel mismatch: expected {self.dim}, got {C}"
        y = x.clone()
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        for i in range(self.num_dilation):
            y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        y3 = self.proj(y2)
        y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
        return y4

class MSDAMLLABlock(nn.Module):
    def __init__(self, c, num_heads_mdla=4, num_heads_mlla=2, shortcut=True):
        super().__init__()
        self.mdla = MultiDilatelocalAttention(c, num_heads=num_heads_mdla)
        self.mlla = MLLABlock(c, num_heads=num_heads_mlla)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        x = self.mdla(x)
        if self.shortcut:
            x = x + residual
        residual = x
        x = self.mlla(x)
        if self.shortcut:
            x = x + residual
        return x

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class RepCSP_MSDAMLLA(RepCSP):
    """RepCSP变体，使用MSDAMLLABlock替换RepBottleneck。"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSDAMLLABlock(c_, shortcut=shortcut) for _ in range(n)))  # 替换RepBottleneck为MSDAMLLABlock

class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class RepNCSPELAN4_MSDAMLLA_bottleneck(RepNCSPELAN4):
    """RepNCSPELAN4变体，使用RepCSP_MSDAMLLA（内部MSDAMLLABlock替换RepBottleneck）。"""
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n)
        self.cv2 = nn.Sequential(RepCSP_MSDAMLLA(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP_MSDAMLLA(c4, c4, n), Conv(c4, c4, 3, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

if __name__ == "__main__":
    # 测试RepNCSPELAN4_MSDAMLLA_bottleneck（修复类未定义错误）
    image_size = (1, 64, 224, 224)  # 测试方形
    image = torch.rand(*image_size)
    model = RepNCSPELAN4_MSDAMLLA_bottleneck(64, 128, 64, 32, n=1)  # 示例参数
    out = model(image)
    print(out.size())  # (1, 128, 224, 224)

    # 测试矩形
    image_size_rect = (1, 64, 128, 256)
    image_rect = torch.rand(*image_size_rect)
    out_rect = model(image_rect)
    print(out_rect.size())  # (1, 128, 128, 256)