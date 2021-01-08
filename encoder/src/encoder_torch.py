import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils_torch import weights_init
from utils_torch import WNConv2d, \
    WNLinear, \
    AntiAliasDownsampleLayer, \
    FastGlobalAvgPool2d

class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers, spk_dim=128):
        super().__init__()
        self.num_speakers = num_speakers        
        self.spk_dim = spk_dim
        ngf = 32        
        model = []
        model += [            
            WNConv2d(in_channels=1, out_channels=ngf, kernel_size=(9,5), stride=1, padding=0, groups=1),            
            nn.LeakyReLU(0.2, True),
            
            WNConv2d(in_channels=ngf, out_channels=ngf, kernel_size=(9,5), stride=1, padding=0, dilation=2, groups=1),
            nn.LeakyReLU(0.2, True),
            
            WNConv2d(in_channels=ngf, out_channels=ngf, kernel_size=(9,5), stride=1, padding=0, dilation=3, groups=1),
            nn.LeakyReLU(0.2, True),
            ]
        self.conv_stack = nn.Sequential(*model)
        model = [
            WNConv2d(ngf, ngf*2, 3, 1, padding=1),
            AntiAliasDownsampleLayer(channels=ngf*2,  filt_size=3, stride=2),
            
            WNConv2d(ngf*2, ngf*4, 3, 1, padding=1),
            AntiAliasDownsampleLayer(channels=ngf*4,  filt_size=3, stride=2),

            WNConv2d(ngf*4, spk_dim, kernel_size=(3, 1))
        ]
        self.conv_down = nn.Sequential(*model)

        model = [
            FastGlobalAvgPool2d(flatten=True),            
            WNLinear(spk_dim, spk_dim),            
            nn.LeakyReLU(0.2, True),
            WNLinear(spk_dim, spk_dim),            
            nn.LeakyReLU(0.2, True),
            WNLinear(spk_dim, spk_dim),
            nn.LeakyReLU(0.2, True),
            WNLinear(spk_dim, spk_dim),
        ]        
        self.dense = nn.Sequential(*model)
        self.fc = nn.Linear(spk_dim, num_speakers)        
        self.apply(weights_init)    

    def forward(self, x):        
        y = self.conv_stack(x.unsqueeze(1))
        y = self.conv_down(y)
        print(y.shape)
        emb_spk = self.dense(y)
        emb_spk = F.normalize(emb_spk, p=2, dim=1)
        log_p_s_x = self.fc(emb_spk)
        return emb_spk, log_p_s_x

if __name__ == "__main__":
    b = 4
    x = torch.randn(b, 80, 32)
    S = SpeakerEncoder(2)
    e, s = S(x)
    print(s.shape, e.shape)
    # torch.save(S, 't_model.pt')

    # torch.onnx.export(
    #     S,
    #     x,
    #     't_model.onnx',
    #     opset_version=12,
    #     do_constant_folding=True,
    #     keep_initializers_as_inputs=True

    # )
    # x = torch.Tensor([[1, -6, 7, 4], [2, 6, -12, 0]])
    # y = F.normalize(x, p=2, dim=1)

