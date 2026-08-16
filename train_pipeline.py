import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入你的模块
from models.light_roadsam import LightRoadSAM
from data.dataset import SPISSDataset
from utils.loss import calc_seg_loss

def train_step(model, images, gt_masks, gt_boxes_list, device):
    B = images.shape[0]
    
    # 1. 特征提取
    img_embeddings, multi_scale_features = model.image_encoder(images)
    fpn_features = model.adapter_fpn(multi_scale_features)
    
    # 2. 预测框预测
    pred_boxes, pred_scores = model.prompt_generator(fpn_features[-1])

    batch_low_res_masks = []
    total_loss_bbox = torch.tensor(0.0, device=device)

    for i in range(B):
        use_gt = torch.rand(1).item() > 0.5
        if use_gt and len(gt_boxes_list[i]) > 0:
            input_boxes = gt_boxes_list[i].to(device)
        else:
            input_boxes = pred_boxes[i]

        if input_boxes.ndim == 1:
            input_boxes = input_boxes.unsqueeze(0)
        
        N = input_boxes.shape[0]
        if N == 0:
            # 如果没目标，补充一张全黑掩码，保持形状为 [1, 1, 256, 256] 以便后续 cat
            batch_low_res_masks.append(torch.zeros((1, 1, 256, 256), device=device))
            continue

        # 3. Prompt Encoding
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, 
            boxes=input_boxes.float(), 
            masks=None
        )

        # 4. Mask Decoding
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=img_embeddings[i:i+1],
            image_pe=model.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 5. 合并 Mask
        # low_res_masks 形状为 [N, 1, 256, 256]，取 max 后为 [1, 1, 256, 256]
        combined_mask, _ = torch.max(low_res_masks, dim=0, keepdim=True)
        batch_low_res_masks.append(combined_mask)

        # 6. 计算 Box 损失 (修复 Warning 的地方)
        if len(gt_boxes_list[i]) > 0:
            # 强制转换为二维 [K, 4] 形状，避免广播错误
            curr_pred_boxes = pred_boxes[i].view(-1, 4) 
            curr_gt_boxes = gt_boxes_list[i].to(device)
            
            # 取两者数量的最小值进行匹配
            num_match = min(curr_pred_boxes.shape[0], curr_gt_boxes.shape[0])
            if num_match > 0:
    # 将像素坐标除以 1024，转换回 0-1 范围计算 Loss
                loss_box_val = F.l1_loss(
                    curr_pred_boxes[:num_match] / 1024.0, 
                    curr_gt_boxes[:num_match] / 1024.0
                )
                total_loss_bbox += loss_box_val

    # 7. 整体损失计算 (修复 Fatal Error 的地方)
    # 使用 cat 拼接 [1, 1, 256, 256] 的列表 -> [B, 1, 256, 256]
    final_pred_masks = torch.cat(batch_low_res_masks, dim=0) 
    
    gt_masks_small = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
    
    loss_mask = calc_seg_loss(final_pred_masks, gt_masks_small)
    loss_box = total_loss_bbox / B
    
    return loss_mask + 0.1 * loss_box

def main():
    # --- 基础配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    batch_size = 2 # 如果 OOM 请设为 1
    lr_encoder = 1e-5 # 蒸馏过的部分用极小学习率微调
    lr_others = 1e-4  # 新组件用标准学习率

    # --- 1. 模型初始化 ---
    model = LightRoadSAM().to(device)
    
    # 注入第一阶段蒸馏的 Encoder 权重
    distill_ckpt = "checkpoints/tiny_vit_distilled_final.pth"
    if os.path.exists(distill_ckpt):
        print(f"💉 正在注入蒸馏权重: {distill_ckpt}")
        model.image_encoder.load_state_dict(torch.load(distill_ckpt, map_location=device))
    else:
        print("⚠️ 未找到蒸馏权重，将从随机初始化开始训练。")

    # --- 2. 差异化学习率设置 ---
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': lr_encoder},
        {'params': model.adapter_fpn.parameters(), 'lr': lr_others},
        {'params': model.prompt_generator.parameters(), 'lr': lr_others},
        {'params': model.mask_decoder.parameters(), 'lr': lr_others}
    ], weight_decay=0.01)

    # --- 3. 数据准备 ---
    dataset = SPISSDataset(root_dir="./dataset", img_size=1024)
    # 使用 zip(*x) 处理不固定数量的 target
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # --- 4. 训练循环 ---
    print(f"🚀 开始全 Pipeline 训练 (Device: {device})")
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for imgs_tuple, targets_tuple in pbar:
            # 数据解包
            images = torch.stack(imgs_tuple).to(device)
            gt_masks = torch.stack([t['masks'] for t in targets_tuple]).to(device)
            gt_boxes = [t['boxes'] for t in targets_tuple]
            
            # 前向 + 损失
            loss = train_step(model, images, gt_masks, gt_boxes, device)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 限制梯度模长，防止震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            

            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 保存每个阶段的权重
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/light_roadsam_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()