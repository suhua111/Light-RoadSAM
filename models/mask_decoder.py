from segment_anything.modeling import MaskDecoder
import torch.nn as nn


class RoadMaskDecoder(MaskDecoder):
    def __init__(self, transformer, transformer_dim, num_multimask_outputs=3, **kwargs):
        # 修改点：使用关键字参数传参 (transformer=..., transformer_dim=...)
        super().__init__(
            transformer=transformer, 
            transformer_dim=transformer_dim, 
            num_multimask_outputs=num_multimask_outputs, 
            **kwargs
        )
        
        # 增加分类分支：对应你数据集中的 5 个障碍物类别 + 背景
        self.cls_head = nn.Linear(transformer_dim, 6)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        # 调用父类 forward 得到 masks 和 iou 预测
        masks, iou_pred = super().forward(
            image_embeddings, 
            image_pe, 
            sparse_prompt_embeddings, 
            dense_prompt_embeddings, 
            multimask_output
        )
        # 这里可以扩展分类逻辑
        return masks, iou_pred