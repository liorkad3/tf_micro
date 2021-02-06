import torch.nn as nn
import torch
from common import ConvBn, DwConvBn

class Block(nn.Module):
    def __init__(self, in_c, out_c, s, res=False):
        super(Block, self).__init__()
        self.res = res
        self.block = nn.Sequential(
            ConvBn(in_c, out_c, 1, s),
            DwConvBn(out_c, 3)
        )

    def forward(self, x):
        y = self.block(x)
        if self.res:
            y = y + x
        return y

class BigBlock(nn.Module):
    def __init__(self, in_c, out_c, n, s):
        super(BigBlock, self).__init__()
        model = []
        for i in range(n):
            if i == 0:
                block = Block(in_c, out_c, s, res=False)
            else:
                block = Block(out_c, out_c, 1, res=True)
            model.append(block)

        self.big_block = nn.Sequential(*model)
    
    def forward(self, x):
        return self.big_block(x)
        

class Backbone(nn.Module):
    def __init__(self, head_ch):
        super(Backbone, self).__init__()

        self.down_settings = [
            # c  n  s   f
            [16, 1, 1, False],
            [24, 2, 2, True],
            [32, 3, 2, True],
            [64, 2, 1, False],
            [96, 4, 2, True],
            [160, 2, 2, False],
            [320, 1, 1, True],
        ]
        feat_channels = [24, 32, 96, 320]
        in_planes = 64

        self.stem = ConvBn(3, in_planes, 3, 2)

        self.blocks = nn.ModuleList()
        in_ch = in_planes
        for settings in self.down_settings:
            self.blocks.append(
                BigBlock(in_ch, settings[0], n=settings[1], s=settings[2])
            )
            in_ch = settings[0]
        
        self.head_convs = nn.ModuleList([
            ConvBn(in_c, head_ch, 1, 1) for in_c in feat_channels
            ])

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.down_settings[i][3]:
                feats.append(
                    self.head_convs[len(feats)](x)
                )
        return feats





