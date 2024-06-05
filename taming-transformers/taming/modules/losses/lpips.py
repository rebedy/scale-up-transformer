"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features. # RGB 3채널이 들어와 64,128,256,512,512로 불어남.
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)  # nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)  # nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)  # nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)  # nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)  # nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")  # ckpt: 'taming/modules/autoencoder/lpips/vgg.pth'
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):  # 둘 다 [B, 3, 256, 256] value -1.0 ~ 1.0
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)  # 둘 다 5개의 tensor를 담고 있는 namedtuple
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2  # diffs에 5개의 tensor가 담김

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]  # tensor[B, 1, 1, 1] 5개를 담고 있는 리스트
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val  # [B, 1, 1, 1]


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # [B, 3, 256, 256]: Conv(3, 64) -> ReLU -> Conv(64, 64) -> ReLU: [B, 64, 256, 256]
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # [B, 64, 256, 256]: MaxPool -> Conv(64, 128) -> ReLU -> Conv(128, 128) -> ReLU: [B, 128, 128, 128]
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # [B, 128, 128, 128]: MaxPool -> Conv(128, 256) -> ReLU -> Conv(256, 256) -> ReLU -> Conv(256, 256) -> ReLU: [B, 256, 64, 64]
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x]) # [B, 256, 64, 64]: MaxPool -> Conv(256, 512) -> ReLU -> Conv(512, 512) -> ReLU -> Conv(512, 512) -> ReLU: [B, 512, 32, 32]
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x]) # [B, 512, 32, 32]: MaxPool -> Conv(512, 512) -> ReLU -> Conv(512, 512) -> ReLU -> Conv(512, 512) -> ReLU: [B, 512, 16, 16]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

