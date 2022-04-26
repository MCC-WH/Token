from collections import OrderedDict
from utils import resnet_block_dilation
import torch.nn as nn
import torchvision
from torch import nn


## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def extract_features_from_e2e(model):
    state_dict_features = OrderedDict()
    for key, value in model['state_dict'].items():
        if key.startswith('features'):
            state_dict_features[key[9:]] = value
    return state_dict_features

class SOABlock(nn.Module):
    def __init__(self, in_ch, k):
        super(SOABlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.g = nn.Sequential(nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)), nn.BatchNorm2d(self.mid_ch), nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

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
        return z

class ResNet(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool):
        super(ResNet, self).__init__()
        net_in = getattr(torchvision.models, name)(pretrained=False)
        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
 
class ResNet_SOA4(nn.Module):
    def __init__(self, name: str, train_backbone: bool, dilation_block5: bool):
        super(ResNet, self).__init__()
        net_in = getattr(torchvision.models, name)(pretrained=False)
        if name.startswith('resnet'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(name))
        features = nn.Sequential(*features)
        self.outputdim_block5 = 2048
        self.outputdim_block4 = 1024
        self.block1 = features[:4]
        self.block2 = features[4]
        self.block3 = features[5]
        self.block4 = features[6]
        self.soa4 = SOABlock(in_ch=1024, k=4)
        self.block5 = features[7]
        if dilation_block5:
            self.block5 = resnet_block_dilation(self.block5, dilation=2)
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.soa4(x)
        x = self.block5(x)
        return x
