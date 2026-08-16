import torchvision.transforms as T
import torch.nn.functional as F

class RoadResize:
    def __init__(self, target_size=1024):
        self.target_size = target_size

    def __call__(self, image, target):
        # 保持比例缩放，长边为 1024
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # 对 Bbox 也要同步缩放
        if "boxes" in target:
            target["boxes"] *= scale
            
        # 填充到 1024x1024 (SAM 要求的固定输入尺寸)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        image = F.pad(torch.from_numpy(image).permute(2,0,1), (0, pad_w, 0, pad_h))
        
        return image, target