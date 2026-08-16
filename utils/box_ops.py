import numpy as np
import cv2
import torch

def mask_to_boxes(mask, color_to_id):
    """
    将彩色语义分割图转换为 Bounding Boxes。
    mask: RGB 格式 [H, W, 3]
    color_to_id: 颜色到类别 ID 的映射字典
    返回: boxes (N, 4) -> [x1, y1, x2, y2], labels (N,)
    """
    boxes = []
    labels = []
    
    for color, class_id in color_to_id.items():
        if class_id == 0: continue  # 忽略背景
        
        # 匹配颜色提取二值图 (BGR 转 RGB 处理)
        match = np.all(mask == np.array(color), axis=-1)
        binary_mask = match.astype(np.uint8) * 255
        
        # 连通域分析
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 40: continue  # 过滤噪声
            boxes.append([x, y, x + w, y + h])
            labels.append(class_id)
            
    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))
        
    return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.int64)