import torch.nn as nn
import torch
from common import ConvBn, DwConvBn, NonLocalBlockND, dist2
from backbone import Backbone

class Student(Backbone):
    def __init__(self, head_ch, t_head_ch):
        super().__init__(head_ch)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(head_ch, t_head_ch),
            nn.Linear(head_ch, t_head_ch),
            nn.Linear(head_ch, t_head_ch),
            nn.Linear(head_ch, t_head_ch)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(head_ch, t_head_ch, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(head_ch, t_head_ch, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(head_ch, t_head_ch, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(head_ch, t_head_ch, kernel_size=1, stride=1, padding=0),
            ])

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(adoption_channels=32, in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(adoption_channels=32, in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(adoption_channels=32, in_channels=256),
                NonLocalBlockND(adoption_channels=32, in_channels=256),
            ]
        )

        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
            ]
        )

        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])


    def forward(self, x, t_feats):
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        losses={}
        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0

        x = super().forward(x)

        for _i in range(len(x)):
            t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
            size = t_attention_mask.size()
            t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
            t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
            t_attention_mask = t_attention_mask.view(size)

            s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
            size = s_attention_mask.size()
            s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
            s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
            s_attention_mask = s_attention_mask.view(size)

            c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_t_attention_mask.size()
            c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
            c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            c_s_attention_adapted = self.channel_wise_adaptation[_i](
                torch.mean(torch.abs(x[_i]), [2, 3],))
            c_s_attention_adapted = c_s_attention_adapted.view(x[0].size(0), -1, 1, 1)
            # c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_s_attention_adapted.size()
            c_s_attention_adapted = c_s_attention_adapted.view(x[0].size(0), -1)  # 2 x 256
            c_s_attention_mask = torch.softmax(c_s_attention_adapted / c_t, dim=1) * 256
            c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()


            c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()

            kd_feat_loss += dist2(
                t_feats[_i],
                 self.adaptation_layers[_i](x[_i]),
                 attention_mask=sum_attention_mask,
                 channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6

            kd_channel_loss += torch.dist(
                torch.mean(torch.abs(t_feats[_i]), [2, 3]),
                c_s_attention_adapted) * 4e-3 * 6

            t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                t_feats[_i].size(3))
            s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                            x[_i].size(3))
            kd_spatial_loss += torch.dist(
                t_spatial_pool,
                self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        kd_nonlocal_loss = 0
        for _i in range(len(x)):
            s_relation = self.student_non_local[_i](x[_i])
            t_relation = self.teacher_non_local[_i](t_feats[_i])
            kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        losses.update(kd_nonlocal_loss=kd_nonlocal_loss * 7e-5 * 6)
        return losses



