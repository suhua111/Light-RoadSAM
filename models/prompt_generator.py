import torch
import torch.nn as nn

class PromptGenerator(nn.Module):
    def __init__(self, input_dim=256, num_objects=20):
        super().__init__()
        self.num_objects = num_objects
        # 一个简单的回归头，预测每个物体的 [x1, y1, x2, y2]
        # 这里为了简化采用固定数量预测（类似 DETR），或使用全局池化后回归
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_objects * 4),
            nn.Sigmoid() # 归一化到 0-1
        )

    def forward(self, p4_feat):
        # p4_feat: [B, 256, 64, 64]
        batch_size = p4_feat.shape[0]
        # 预测 Bboxes 并 Reshape 为 [B, N, 4]
        bboxes = self.bbox_head(p4_feat).view(batch_size, self.num_objects, 4)
        # 映射回 1024 坐标系
        return bboxes * 1024