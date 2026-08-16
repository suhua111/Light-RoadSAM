import numpy as np
import torch

def calculate_miou(pred_mask, gt_mask, threshold=0.5):
    """
    计算 Batch 的平均 IoU
    pred_mask: [B, 1, H, W] (Logits)
    gt_mask: [B, 1, H, W] (Ground Truth)
    """
    # 将输出转为 0/1 掩码
    pred = (torch.sigmoid(pred_mask) > threshold).float()
    
    # 计算交集和并集
    intersection = (pred * gt_mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3)) - intersection
    
    # 加上 1e-6 防止除以 0
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_iou(preds, labels):
    """
    计算二值掩码的 IoU
    preds: [B, H, W] (0 or 1)
    labels: [B, H, W] (0 or 1)
    """
    intersection = (preds & labels).sum().float()
    union = (preds | labels).sum().float()
    
    if union == 0:
        return 1.0  # 如果两图都为空，认为完全重合
    return (intersection / union).item()

def evaluate_performance(model, dataloader, device):
    model.eval()
    ious = []
    latencies = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            gt_masks = targets['masks'].to(device) # [B, H, W]

            # 测量推理时间
            start_event.record()
            
            # 1. 前向传播：编码器 + 自动提示 + 解码器
            # 这里的 predict 应该包含完整的 pipeline
            pred_masks, _ = model.predict(images) 
            
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

            # 2. 计算 IoU
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).int()
            iou = compute_iou(pred_binary, gt_masks.int())
            ious.append(iou)

    avg_iou = np.mean(ious)
    fps = 1000 / np.mean(latencies) # ms 转 FPS
    return avg_iou, fps
def compute_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    计算检测框的 Average Precision (AP) 的简化版
    pred_boxes: [N, 4] 预测框 (x1, y1, x2, y2)
    gt_boxes: [M, 4] 真值框
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    # 计算预测框与真值框之间的 IoU 矩阵 [N, M]
    ious = _box_iou(pred_boxes, gt_boxes)
    
    # 简单的匹配逻辑：每个预测框如果与某个 GT 的 IoU 大于阈值，则视为 TP
    max_ious, _ = ious.max(dim=1)
    tp = (max_ious > iou_threshold).float()
    
    # 返回精度（由于你的场景主要是针对障碍物存在与否，这里计算 Precision）
    return tp.mean().item()

def _box_iou(boxes1, boxes2):
    """计算两组框之间的 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / union