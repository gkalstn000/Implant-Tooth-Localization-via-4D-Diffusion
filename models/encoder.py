import torch
from .nn import (conv_nd, linear, normalization, timestep_embedding,
                 torch_checkpoint, zero_module)
from torch.nn import init
import torch.nn as nn
from .blocks import Downsample, SpatioTemporalAttentionWithAbsolutePosition

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = conv_nd(3, in_channels, out_channels, 3)
        self.bn1 = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_nd(3, out_channels, out_channels, 3)
        self.bn2 = normalization(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv_nd(3, in_channels, out_channels, 1),
                normalization(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
class Encoder(nn.Module) :
    def __init__(self, conf):
        super(Encoder, self).__init__()
        self.conf = conf
        self.layer1 = nn.Sequential(BasicResBlock(1, 32),
                                    Downsample(32, False, 3))
        self.layer2 = nn.Sequential(BasicResBlock(32, 128),
                                    Downsample(128, False, 3))
        self.layer3 = nn.Sequential(BasicResBlock(128, 128),
                                    SpatioTemporalAttentionWithAbsolutePosition(
                                        dim=128,
                                        dim_head=128 // conf.attn_numheads,
                                        heads=conf.attn_numheads,
                                        max_frames=self.conf.maxframe,
                                        sub_frame=self.conf.sub_frame,
                                        resolution=32
                                    ),
                                    Downsample(128, False, 3)
                                    )

        self.layer4 = nn.Sequential(BasicResBlock(128, 64),
                                    SpatioTemporalAttentionWithAbsolutePosition(
                                        dim=64,
                                        dim_head=64 // conf.attn_numheads,
                                        heads=conf.attn_numheads,
                                        max_frames=self.conf.maxframe,
                                        sub_frame=self.conf.sub_frame,
                                        resolution=16
                                    ),
                                    Downsample(64, False, 3)
                                    )

        self.layer5 = nn.Sequential(BasicResBlock(64, 32),
                                    SpatioTemporalAttentionWithAbsolutePosition(
                                        dim=32,
                                        dim_head=32 // conf.attn_numheads,
                                        heads=conf.attn_numheads,
                                        max_frames=self.conf.maxframe,
                                        sub_frame=self.conf.sub_frame,
                                        resolution=8
                                    ),
                                    Downsample(32, False, 3)

                                    )

        self.out_fc = nn.Sequential(
            nn.Linear(4*4*self.conf.sub_frame*32, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        b, c, f, h, w = x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(b, -1)
        return torch.sigmoid(self.out_fc(x))
    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)