import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class TinyViTEncoder(nn.Module):
    def __init__(self, model_name='tiny_vit_5m_224', pretrained=False, checkpoint_path=None):
        super().__init__()
        # 1. 创建模型架构 (离线模式不联网)
        self.backbone = timm.create_model(model_name, pretrained=False, features_only=True)
        
        # 2. 手动加载本地预训练权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"==> 正在从本地加载 Tiny-ViT 预训练权重: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理 timm/huggingface 常见的嵌套格式
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # 过滤掉不匹配的权重键值 (features_only 模式下有些头会被丢弃)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"==> 权重加载完成。提示信息: {msg}")
        elif pretrained:
            print("==> 警告: 设置了 pretrained=True 但未提供本地路径，且服务器断网，加载将跳过。")

        # 3. 特征对齐 Neck
        # 获取 Tiny-ViT 最后一层通道数 (Tiny-ViT-5M 通常是 448)
        self.feature_info = self.backbone.feature_info
        last_ch = self.feature_info[-1]['num_chs']
        
        self.neck = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            # 上采样对齐到 SAM Decoder 需要的 64x64 特征图
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, x):
        # 提取 Stage 1-4 特征
        features = self.backbone(x)
        # 将 Stage 4 特征通过 Neck 映射到 SAM 空间
        img_embeddings = self.neck(features[-1]) 
        return img_embeddings, features