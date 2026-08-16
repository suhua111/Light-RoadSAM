import torch.nn.functional as F
import torch

def calc_seg_loss(pred_mask, gt_mask):
    # 必须先经过 sigmoid 将 logits 转为 0-1 的概率
    pred_prob = torch.sigmoid(pred_mask) 
    
    # Dice Loss 使用概率值计算
    num = 2 * (pred_prob * gt_mask).sum()
    den = pred_prob.sum() + gt_mask.sum()
    dice_loss = 1 - (num + 1e-5) / (den + 1e-5)
    
    # BCE 继续使用 logits (因为函数内部自带 sigmoid)
    bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    
    return 0.5 * dice_loss + 0.5 * bce_loss

def iou_loss(pred_boxes, gt_boxes):
    """计算预测框与真实框的交并比损失"""
    # 转换坐标格式并计算交集
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_p + area_g - inter
    
    iou = inter / (union + 1e-6)
    return 1 - iou.mean()