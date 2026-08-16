import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterFPN(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 448], out_channels=256):
        super().__init__()
        # 1x1 卷积将不同层级的通道数对齐到 256
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        # 3x3 卷积消除上采样带来的混叠
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, x_list):
        # x_list 包含从浅到深的 4 个特征图
        # 自顶向下融合 (Top-Down)
        last_inner = self.lateral_convs[-1](x_list[-1])
        results = [last_inner]
        
        for i in range(len(x_list) - 2, -1, -1):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            lateral_input = self.lateral_convs[i](x_list[i])
            last_inner = inner_top_down + lateral_input
            results.insert(0, last_inner)
            
        # 最终平滑处理
        fpn_outs = [self.output_convs[i](results[i]) for i in range(len(results))]
        return fpn_outs # 返回 [P1, P2, P3, P4]