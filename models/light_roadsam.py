import torch
import torch.nn as nn
from .tiny_vit import TinyViTEncoder
from .adapter_fpn import AdapterFPN
from .prompt_generator import PromptGenerator
from .mask_decoder import RoadMaskDecoder
from segment_anything.modeling import PromptEncoder, TwoWayTransformer

class LightRoadSAM(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Image Encoder
        self.image_encoder = TinyViTEncoder()
        
        # 2. Adapter FPN
        # --- 核心修正点：根据报错修改通道数 ---
        # 报错说拿到了 320，说明 TinyViT-5M 的最后一层是 320
        # 标准 TinyViT-5M 的四层通道通常是 [64, 128, 160, 320]
        self.adapter_fpn = AdapterFPN(in_channels_list=[64, 128, 160, 320])
        
        # 3. Prompt Generator
        self.prompt_generator = PromptGenerator()
        
        # 4. SAM 标准组件
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        
        self.mask_decoder = RoadMaskDecoder(
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256
        )

    def get_dense_pe(self):
        """获取位置编码"""
        return self.prompt_encoder.get_dense_pe()

    def predict_prompts(self, x):
        """供推理和训练逻辑调用"""
        img_embeddings, features = self.image_encoder(x)
        fpn_feats = self.adapter_fpn(features)
        # 使用 FPN 融合后的最后一层特征进行框预测
        pred_boxes, pred_scores = self.prompt_generator(fpn_feats[-1])
        return img_embeddings, pred_boxes

    def forward(self, x):
        img_embeddings, pred_boxes = self.predict_prompts(x)
        # 默认前向传播（如需在推理中使用）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=pred_boxes, masks=None
        )
        masks, _ = self.mask_decoder(
            image_embeddings=img_embeddings,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return masks, pred_boxes