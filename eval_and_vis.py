import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from models.light_roadsam import LightRoadSAM
from data.dataset import SPISSDataset
import torch.nn.functional as F
import os
from tqdm import tqdm

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型与权重
    print("🔄 正在初始化模型并加载权重...")
    model = LightRoadSAM().to(device)
    ckpt_path = "checkpoints/light_roadsam_epoch_30.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ 未找到权重文件: {ckpt_path}，请确认路径是否正确。")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 2. 准备测试数据
    dataset = SPISSDataset(root_dir="./dataset", img_size=1024)
    # shuffle=False 保证结果可追溯
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    iou_list = []
    # 确保输出文件夹干净，方便查看新结果
    save_dir = "./eval_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"📂 结果将保存至 {save_dir}")

    print(f"🔍 开始全量评估 (总计样本数: {len(dataset)})...")
    
    with torch.no_grad():
        # 使用 tqdm 监控进度，它会显示在你的终端界面上
        for i, (imgs, targets) in enumerate(tqdm(test_loader, desc="Eval Progress")):
            
            image_tensor = torch.stack(imgs).to(device)
            # gt_mask 形状 [1, 1, 1024, 1024]
            gt_mask = targets[0]['masks'].to(device).unsqueeze(0) 
            
            # --- 推理流程 ---
            # 1. 特征提取
            img_embed, multi_feats = model.image_encoder(image_tensor)
            fpn_feats = model.adapter_fpn(multi_feats)
            
            # 2. 预测候选框
            pred_boxes = model.prompt_generator(fpn_feats[-1])
            
            # 3. Prompt Encoding (处理 20 个候选框)
            sparse, dense = model.prompt_encoder(
                points=None, 
                boxes=pred_boxes[0].float(), 
                masks=None
            )
            
            # 4. Mask Decoding
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )
            
            # 5. 合并 Mask (取 20 个预测的最大并集)
            combined_mask_low, _ = torch.max(low_res_masks, dim=0, keepdim=True)
            
            # 6. 还原到原图尺寸并二值化
            pred_mask_logits = F.interpolate(combined_mask_low, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).float()

            # --- 指标计算 ---
            inter = (pred_mask * gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum() - inter
            iou = (inter + 1e-6) / (union + 1e-6)
            iou_list.append(iou.item())

            # --- 保存可视化结果 ---
            # 为了避免硬盘瞬间塞满，我们这里保存前 200 张图片（你可以根据需要修改这个数字）
            if i < 200:
                vis_img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
                # 反归一化
                vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                
                # 绿色填充预测区域 (Semi-transparent)
                mask_np = pred_mask[0, 0].cpu().numpy()
                vis_img[mask_np > 0] = vis_img[mask_np > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
                
                # 红色线条勾勒真实轮廓 (Ground Truth)
                gt_np = gt_mask[0, 0].cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
                
                # 在图上写上 IoU 分数
                cv2.putText(vis_img, f"IoU: {iou.item():.4f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imwrite(os.path.join(save_dir, f"sample_{i:04d}_iou_{iou.item():.2f}.jpg"), vis_img)

    print("-" * 30)
    print(f"📊 评估任务顺利完成!")
    print(f"📈 全量 mIoU (Mean Intersection over Union): {np.mean(iou_list):.4f}")
    print(f"✅ 可视化对比图已保存至: {os.path.abspath(save_dir)}")
    print("💡 提示：你可以使用 'ls -l eval_results/' 命令查看生成的文件。")

if __name__ == "__main__":
    evaluate()