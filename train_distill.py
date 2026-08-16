import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 修改点 1: 适配新版 GradScaler 导入路径
from torch.amp import autocast, GradScaler 

from segment_anything import sam_model_registry
from models.tiny_vit import TinyViTEncoder
from data.dataset import SPISSDataset 

def distillation_loss(student_feat, teacher_feat):
    """
    多准则蒸馏损失函数
    """
    # 1. MSE Loss: 强制像素级数值对齐
    mse_loss = F.mse_loss(student_feat, teacher_feat)
    
    # 2. Cosine Similarity Loss: 强化语义特征的方向一致性
    cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=1)
    cos_loss = (1 - cos_sim).mean()
    
    # 权重分配：数值对齐 (1.0) + 语义对齐 (2.0)
    return mse_loss + 2.0 * cos_loss

# --- 新增修改点：自定义 Collate Function ---
def distill_collate_fn(batch):
    """
    处理 batch 中由于目标数量不同导致的维度不一致问题。
    images: 堆叠为 [B, 3, 1024, 1024]
    targets: 保持为 List 格式
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

def train_distill(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 修改点 2: 消除 FutureWarning，明确指定 cuda
    scaler = GradScaler('cuda') 
    
    print("Loading Teacher: SAM ViT-H...")
    sam_teacher = sam_model_registry["vit_h"](checkpoint=args.teacher_ckpt)
    teacher_encoder = sam_teacher.image_encoder.to(device)
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False
    
    print("Initializing Student: Tiny-ViT...")
    # 修改点 3: 增加本地权重路径参数
    student = TinyViTEncoder(
        model_name=args.student_model_name,
        pretrained=False, # 离线环境设为 False
        checkpoint_path=args.student_pretrain_ckpt # 传入本地权重
    ).to(device)
    student.train()
    
    # --- 修改点：在 DataLoader 中应用 distill_collate_fn ---
    dataset = SPISSDataset(root_dir=args.data_root) 
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=distill_collate_fn  # 解决维度不一致导致的 Resize 报错
    )
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Starting Distillation for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0
        
        # 修改点：明确解包 images 和 targets (虽然蒸馏暂不用 targets)
        for images, targets in pbar:
            images = images.to(device) 
            
            # 修改点 4: 适配新版 autocast 写法
            with autocast('cuda'):
                with torch.no_grad():
                    teacher_feat = teacher_encoder(images) # [B, 256, 64, 64]
                
                # Student 前向
                student_feat, _ = student(images) 
                
                # 计算蒸馏损失
                loss = distillation_loss(student_feat, teacher_feat)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
        scheduler.step()
        
        # 定期保存权重
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(student.state_dict(), f"checkpoints/tiny_vit_distill_epoch_{epoch+1}.pth")

    torch.save(student.state_dict(), "checkpoints/tiny_vit_distilled_final.pth")
    print("Distillation Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--student_model_name", type=str, default="tiny_vit_5m_224")
    parser.add_argument("--student_pretrain_ckpt", type=str, default="checkpoints/tiny_vit_5m_224.pth")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    train_distill(args)