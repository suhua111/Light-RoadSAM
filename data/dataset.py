import os
from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.box_ops import mask_to_boxes

class SPISSDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=1024):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.img_size = img_size 
        self.color_to_id = {
            (0, 0, 0): 0, (0, 0, 255): 1, (0, 255, 255): 2,
            (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5,
        }
        paths = list(self.root_dir.rglob("img/*.png"))
        self.img_paths = sorted([str(p) for p in paths])
        
        # 调试：确保初始化时能看到路径数量
        print(f"✅ 数据集初始化完成，找到样本: {len(self.img_paths)}")

    # --- 核心修复点：确保 __len__ 与 __init__ 对齐 ---
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = img_path.replace("/img/", "/seg/")
        
        # 读取图像并缩放
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # 读取 Mask
        mask_color = cv2.imread(mask_path)
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
        
        # --- 补全逻辑：生成全局语义掩码 (KeyError: 'masks' 的补救) ---
        # 创建一个全黑的掩码，将所有障碍物颜色变为 1
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        for color, cls_id in self.color_to_id.items():
            if cls_id == 0: continue
            match = np.all(mask_color == color, axis=-1)
            semantic_mask[match] = 1
            
        mask_resized = cv2.resize(semantic_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() # [1, 1024, 1024]
        
        # 获取 Boxes 并缩放坐标
        boxes, labels = mask_to_boxes(mask_color, self.color_to_id)
        sx, sy = self.img_size / w, self.img_size / h
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # 转图像为 Tensor
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": mask_tensor, # 修复之前的 KeyError
            "orig_size": torch.tensor([h, w])
        }

        return image_tensor, target