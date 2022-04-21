import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
from .backbone import ResNet

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor):
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor):
        return drop_path(x, self.drop_prob, self.training)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        nn.init.constant_(self.proj.weight.data, 0.0)
        nn.init.constant_(self.proj.bias.data, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        B_q, N_q, _ = q.size()
        B_k, N_k, _ = k.size()
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        q = (attn @ v).transpose(1, 2).reshape(q.size(0), q.size(2), -1)
        q = self.proj_drop(self.proj(q))
        return q


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn = nn.BatchNorm1d(dim)
        self.mlp = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, d = x.size()
        x = x + self.drop_path(self.attn(x, x, x))
        x_bn = self.bn(x.reshape(b * n, d)).reshape(b, n, d)
        x = x + self.drop_path(self.mlp(x_bn))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
        return q


class Token_Refine(nn.Module):
    def __init__(self, num_heads, num_object, mid_dim=1024, encoder_layer=1, decoder_layer=2, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.query = Parameter(torch.randn(1, num_object, mid_dim))
        self.token_norm = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.LayerNorm(mid_dim))
        self.encoder = nn.ModuleList([Encoder(mid_dim, num_heads, qkv_bias, drop, attn_drop, drop_path) for _ in range(encoder_layer)])
        self.decoder = nn.ModuleList([Decoder(mid_dim, num_heads, qkv_bias, drop, attn_drop, drop_path) for _ in range(decoder_layer)])
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=mid_dim, kernel_size=(1, 1), stride=1, padding=0), nn.BatchNorm2d(mid_dim))
        self.mid_dim = mid_dim
        self.proj = nn.Sequential(nn.Linear(in_features=mid_dim * num_object, out_features=1024), nn.BatchNorm1d(1024))

    def forward(self, x: Tensor):
        B, _, H, W = x.size()
        x = self.conv(x).reshape(B, self.mid_dim, H * W).permute(0, 2, 1)
        for encoder in self.encoder:
            x = encoder(x)
        q = self.query.repeat(B, 1, 1)  # B x num_object x mid_dim
        attns = F.softmax(torch.bmm(q, x.permute(0, 2, 1)), dim=1)  # b x num_object x (H x W)
        token = torch.bmm(attns, x)
        token = self.token_norm(token)
        for decoder in self.decoder:
            token = decoder(token, x)
        token = self.proj(token.reshape(B, -1))
        return token


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, eps=1e-6):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.threshold = math.pi - self.m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, dim=-1), F.normalize(self.weight, dim=-1))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros(cos_theta.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot).bool()

        output = torch.cos(torch.where(selected, theta + self.m, theta))
        output *= self.s
        return output


class RetrievalNet(nn.Module):
    def __init__(self, classifier_num):
        super().__init__()
        outputdim = 1024
        self.outputdim = 1024
        self.backbone = ResNet(name='resnet101', train_backbone=True, dilation_block5=False)
        self.tr = Token_Refine(num_heads=8, num_object=4, mid_dim=outputdim, encoder_layer=1, decoder_layer=2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.tr(x)
        global_feature = F.normalize(x, dim=-1)
        return global_feature
