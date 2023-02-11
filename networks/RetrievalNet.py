import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
from .backbone import ResNet, ResNet_STAGE45, weights_init, constant_init

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
        self.query = nn.Parameter(torch.randn(1, num_object, mid_dim))
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


#----------------------------------------------------------------------------------------------------------------------------------
# Token
class Token(nn.Module):
    def __init__(self, outputdim=1024, classifier_num=81313):
        super().__init__()
        self.outputdim = 1024
        self.backbone = ResNet(name='resnet101', train_backbone=True, dilation_block5=False)
        self.tr = Token_Refine(num_heads=8, num_object=4, mid_dim=outputdim, encoder_layer=1, decoder_layer=2)
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.tr(x)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    def forward(self, x, label):
        x = self.backbone(x)
        x = self.tr(x)
        global_logits = self.classifier(x, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits


#----------------------------------------------------------------------------------------------------------------------------------
# RMAC


class rmac(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super(rmac, self).__init__()

    def forward(self, x):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)

        b = (max(H, W) - w) / (steps - 1)
        (_, idx) = torch.min(torch.abs(((w**2 - w * b) / w**2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + self.eps).expand_as(v)

        for l in range(1, self.L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + self.eps).expand_as(vt)
                    v += vt

        return v


class RMAC(nn.Module):
    def __init__(self, outputdim, classifier_num):
        super(RMAC, self).__init__()
        self.backbone = ResNet(name='resnet101', train_backbone=True, dilation_block5=False)
        self.pooling = rmac()
        self.whiten = nn.Conv2d(self.backbone.outputdim_block5, 2048, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = outputdim
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.whiten(x).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    def forward(self, x, label):
        x = self.backbone(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(x, p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits


#----------------------------------------------------------------------------------------------------------------------------------
# DELG


class Spatial_Attention(nn.Module):
    def __init__(self, input_dim):
        super(Spatial_Attention, self).__init__()
        self.att_conv1 = nn.Conv2d(input_dim, 1, kernel_size=(1, 1), padding=0, stride=1, bias=False)
        self.att_act2 = nn.Softplus(beta=1, threshold=20)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        att_score = self.att_act2(self.att_conv1(x))
        return att_score


class DELG(nn.Module):
    def __init__(self, outputdim, reduction_dim=128, classifier_num=1024):
        super(DELG, self).__init__()

        self.backbone = ResNet_STAGE45(name='resnet101', train_backbone=True, dilation_block5=False)
        self.pooling = gem(p=3.0)
        self.whiten = nn.Conv2d(self.backbone.outputdim_block5, outputdim, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = outputdim
        self.classifier_block5 = ArcFace(in_features=outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)
        self.classifier_block4 = ArcFace(in_features=reduction_dim, out_features=classifier_num, s=math.sqrt(reduction_dim), m=0.1)
        self.attention = Spatial_Attention(input_dim=1024)
        self.reduction = nn.Conv2d(self.backbone.outputdim_block4, reduction_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def _init_input_proj(self, weight, bias):
        self.reduction.weight.data = weight
        self.reduction.bias.data = bias

    @torch.no_grad()
    def forward_test(self, x):
        x4, x5 = self.backbone(x)
        global_feature = F.normalize(self.pooling(x5), p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(global_feature, p=2.0, dim=-1)
        return global_feature

    def forward(self, x, label):
        x4, x5 = self.backbone(x)
        global_feature = F.normalize(self.pooling(x5), p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_logits = self.classifier_block5(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)

        x4 = x4.detach()
        att_score = self.attention(x4)
        local_features = F.normalize(self.reduction(x4), dim=-1)
        att_local_features = local_features * att_score
        pooled_local_feature = F.avg_pool2d(att_local_features, (att_local_features.size(-2), att_local_features.size(-1))).squeeze(-1).squeeze(-1)
        local_logits = self.classifier_block4(pooled_local_feature, label)
        local_loss = F.cross_entropy(local_logits, label)
        return global_loss, local_loss, global_logits, local_logits


#----------------------------------------------------------------------------------------------------------------------------------
# SOLAR


class gem(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(gem, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class SOABlock_GeM(nn.Module):
    def __init__(self, in_ch, k):
        super(SOABlock_GeM, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.g = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))
        self.pooling = gem(p=3.0)

        for conv in [self.f, self.g, self.h]:
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def forward(self, x: Tensor):
        B, C, H, W = x.size()

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W
        attn = F.softmax((self.mid_ch**-.50) * z, dim=-1)
        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W
        z = self.v(z)
        z = z + x
        z = self.pooling(z)
        return z


class SOLAR(nn.Module):
    def __init__(self, outputdim, classifier_num):
        super(SOLAR, self).__init__()
        self.backbone = ResNet(name='resnet101', train_backbone=True, dilation_block5=False)
        self.pooling = SOABlock_GeM(in_ch=2048, k=2)
        self.whiten = nn.Conv2d(self.backbone.outputdim_block5, 2048, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.outputdim = outputdim
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = F.normalize(x, p=2.0, dim=1)
        x = self.whiten(x).squeeze(-1).squeeze(-1)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    def forward(self, x, label):
        x = self.backbone(x)
        x = self.pooling(x)
        global_feature = F.normalize(x, p=2.0, dim=1)
        global_feature = self.whiten(global_feature).squeeze(-1).squeeze(-1)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits


#----------------------------------------------------------------------------------------------------------------------------------
# NetVLAD


class VLADLayer(nn.Module):
    def __init__(self, num_clusters=64, dim=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.encoder = Encoder(dim, 8, True, drop=0.1, attn_drop=0.1, drop_path=0.1)
        self.centroids = nn.Parameter(torch.randn(1, dim, num_clusters))
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=dim, kernel_size=(1, 1), stride=1, padding=0), nn.BatchNorm2d(dim))
        self.mid_dim = dim
        self.proj = nn.Sequential(nn.Linear(in_features=dim * num_clusters, out_features=dim), nn.BatchNorm1d(dim))

    def forward(self, x: Tensor):
        N, _, H, W = x.size()
        x = self.conv(x).reshape(N, self.dim, H * W).permute(0, 2, 1)
        x = self.encoder(x)

        # soft-assignment
        centroids = self.centroids.repeat(N, 1, 1)
        soft_assign = F.softmax(torch.bmm(x, centroids), dim=-1)  # (N, H * W, number_cluster)

        residual = x.unsqueeze(2) - centroids.permute(0, 2, 1).unsqueeze(1)  # (N, H * W, number_cluster, C)
        vlad = torch.sum(residual * soft_assign.unsqueeze(-1), dim=1)  # (N, number_cluster, C)
        norm_vlad = F.normalize(vlad, dim=-1)  # intra-normalization
        cos_cluster = torch.bmm(norm_vlad, norm_vlad.permute(0, 2, 1))
        mask = torch.scatter(torch.ones_like(cos_cluster), -1, torch.arange(cos_cluster.size(1), device=cos_cluster.device).view(-1, 1).repeat(N, 1, 1), 0.0)
        Regularization = torch.mean(cos_cluster * mask)
        VLAD = self.proj(vlad.reshape(N, -1))
        return VLAD, Regularization


class NetVLAD(nn.Module):
    def __init__(self, outputdim, classifier_num):
        super().__init__()
        self.outputdim = outputdim
        self.backbone = ResNet(name='resnet101', train_backbone=False, type='imagenet', dilation_block5=False)
        self.vlad = VLADLayer(num_clusters=4, dim=self.outputdim)
        self.classifier = ArcFace(in_features=self.outputdim, out_features=classifier_num, s=math.sqrt(self.outputdim), m=0.2)

    @torch.no_grad()
    def forward_test(self, x):
        x = self.backbone(x)
        x, _ = self.vlad(x)
        global_feature = F.normalize(x, dim=-1)
        return global_feature

    def forward(self, x, label):
        x = self.backbone(x)
        x, _ = self.vlad(x)
        global_feature = F.normalize(x, dim=-1)
        global_logits = self.classifier(global_feature, label)
        global_loss = F.cross_entropy(global_logits, label)
        return global_loss, global_logits
