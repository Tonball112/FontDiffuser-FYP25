import os
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image

# Standard FontDiffuser imports
from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics

def calculate_structural_metrics(pred_tensor, target_tensor):
    def extract_skeleton(img_tensor):
        dilated = F.max_pool2d(img_tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-img_tensor, kernel_size=3, stride=1, padding=1)
        return dilated - eroded
        
    pred_skel = extract_skeleton(pred_tensor)
    target_skel = extract_skeleton(target_tensor)
    skel_l1 = F.l1_loss(pred_skel, target_skel).item()

    threshold = 0.1
    pred_bin = (pred_skel > threshold).float()
    targ_bin = (target_skel > threshold).float()
    intersection = (pred_bin * targ_bin).sum()
    union = pred_bin.sum() + targ_bin.sum() - intersection
    skel_iou = (intersection / union).item() if union > 0 else 1.0

    pred_gray = pred_tensor.mean(dim=1, keepdim=True)
    targ_gray = target_tensor.mean(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred_tensor.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred_tensor.device)
    
    pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
    
    targ_edge_x = F.conv2d(targ_gray, sobel_x, padding=1)
    targ_edge_y = F.conv2d(targ_gray, sobel_y, padding=1)
    targ_edge = torch.sqrt(targ_edge_x**2 + targ_edge_y**2 + 1e-6)
    
    edge_l1 = F.l1_loss(pred_edge, targ_edge).item()
    return skel_l1, skel_iou, edge_l1

def main():
    args = arg_parse()
    pipeline, model_params = load_fontdiffuser_pipeline(args)
    
    # The 4 seen characters for your balanced comparison
    characters = ['帝', '己', '典', '假']
    
    # Exact paths based on your configuration 
    CONTENT_DIR = "data_sxh/train/ContentImage/standard"
    STYLE_IMAGE_PATH = "data_sxh/train/StyleImage/sxh/sxh+中.png" 
    GT_DIR = "data_sxh/seen_ground_truth"
    
    results = {}
    transform = TF.Compose([
        TF.Resize((128, 128)),
        TF.ToTensor(),
        TF.Normalize([0.5], [0.5])
    ])
    
    print("=== STARTING BALANCED SEEN EVALUATION (4 vs 4) ===")
    style_image = Image.open(STYLE_IMAGE_PATH).convert("RGB")
    
    for char in characters:
        print(f"\nProcessing Seen Character: {char}")
        
        content_path = os.path.join(CONTENT_DIR, f"{char}.png")
        gt_path = os.path.join(GT_DIR, f"{char}.png")
        
        if not os.path.exists(gt_path):
            print(f"  Error: Ground truth not found at {gt_path}")
            continue

        content_image = Image.open(content_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        
        # Sampling with LoRA injected 
        pred_image = sampling(
            args=args,
            pipeline=pipeline,
            content_image=content_image,
            style_image=style_image,
            model_params=model_params
        )
        
        pred_tensor = transform(pred_image).unsqueeze(0).to(args.device)
        gt_tensor = transform(gt_image).unsqueeze(0).to(args.device)
        
        # Compute metrics 
        char_metrics = FontMetrics(gt_tensor, pred_tensor)
        scores = char_metrics.compute()
        skel_l1, skel_iou, edge_l1 = calculate_structural_metrics(pred_tensor, gt_tensor)
            
        results[char] = {
            "SSIM": scores["ssim"],
            "L1_Pixel": scores["l1"],
            "L1_Skeleton": skel_l1,
            "IoU_Skeleton": skel_iou,
            "L1_Edge": edge_l1
        }
        
    # Print Balanced Summary
    print("\n=== BALANCED SEEN SUMMARY (4 CHARACTERS) ===")
    print(f"{'Char':<6} | {'SSIM (↑)':<10} | {'L1 Pix(↓)':<10} | {'Skel L1(↓)':<12} | {'Skel IoU(↑)':<12} | {'Edge L1(↓)':<12}")
    print("-" * 75)
    
    metrics_sums = [0.0] * 5
    for char, mets in results.items():
        vals = [mets['SSIM'], mets['L1_Pixel'], mets['L1_Skeleton'], mets['IoU_Skeleton'], mets['L1_Edge']]
        print(f"{char:<6} | {vals[0]:<10.4f} | {vals[1]:<10.4f} | {vals[2]:<12.4f} | {vals[3]:<12.4f} | {vals[4]:<12.4f}")
        for i in range(5): metrics_sums[i] += vals[i]
        
    avg = [v/len(results) for v in metrics_sums]
    print("-" * 75)
    print(f"{'AVG':<6} | {avg[0]:<10.4f} | {avg[1]:<10.4f} | {avg[2]:<12.4f} | {avg[3]:<12.4f} | {avg[4]:<12.4f}")

if __name__ == "__main__":
    main()