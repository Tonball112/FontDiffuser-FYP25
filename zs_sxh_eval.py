# Zero-Shot Specialized Evaluation Script with Structural Metrics

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image

from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics

def calculate_structural_metrics(pred_tensor, target_tensor):
    # 1. Morphological Skeleton L1
    def extract_skeleton(img_tensor):
        dilated = F.max_pool2d(img_tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-img_tensor, kernel_size=3, stride=1, padding=1)
        return dilated - eroded
        
    pred_skel = extract_skeleton(pred_tensor)
    target_skel = extract_skeleton(target_tensor)
    skel_l1 = F.l1_loss(pred_skel, target_skel).item()

    # 2. Skeleton IoU (Intersection over Union)
    threshold = 0.1
    pred_bin = (pred_skel > threshold).float()
    targ_bin = (target_skel > threshold).float()
    intersection = (pred_bin * targ_bin).sum()
    union = pred_bin.sum() + targ_bin.sum() - intersection
    skel_iou = (intersection / union).item() if union > 0 else 1.0

    # 3. Sobel Edge Loss
    pred_gray = pred_tensor.mean(dim=1, keepdim=True)
    targ_gray = target_tensor.mean(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred_tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred_tensor.device).view(1, 1, 3, 3)
    
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
    
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        raise ValueError("You must provide --ckpt_dir in the command line.")
    
    # --- CONFIGURATION ---
    content_dir = "data_sxh/train/ContentImage"
    ground_truth_dir = "data_sxh/zero_shot_ground_truth" 
    style_image_path = "data_sxh/train/TargetImage/sxh/sxh+道.png" 
    
    output_dir = "outputs/eval_results/zero_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    test_chars = ["丙", "夔", "了", "互"]
    
    # Setup Pipeline
    args.guidance_type = "classifier-free"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.guidance_scale = 7.5
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"
    args.demo = True
    args.num_inference_steps = 20
    args.batch_size = 1
    args.seed = 1234 
    
    pipe = load_fontdiffuser_pipeline(args=args)
    toTensor = TF.ToTensor()
    style_image = Image.open(style_image_path).convert("RGB")

    print("\n=== STARTING ZERO-SHOT EVALUATION ===")
    
    results = {}
    
    for char in test_chars:
        print(f"\nEvaluating: {char}")
        
        content_img = Image.open(f"{content_dir}/{char}.png").convert("RGB")
        sampling_args = dict(args=args, pipe=pipe, content_image=content_img, style_image=style_image)
        out_image = sampling(**sampling_args)
        out_image.save(f"{output_dir}/generated_{char}.png")
        
        gt_path = f"{ground_truth_dir}/{char}.png"
        if not os.path.exists(gt_path):
            print(f"[ERROR] Ground truth not found at {gt_path}. Cannot compute metrics for {char}!")
            continue
            
        gt_image = Image.open(gt_path).convert("RGB")
        if gt_image.size != (96, 96):
            gt_image = gt_image.resize((96, 96), Image.Resampling.BILINEAR)
            
        char_metrics = FontMetrics(device=args.device)
        pred_tensor = torch.stack([toTensor(out_image)]).to(args.device)
        gt_tensor = torch.stack([toTensor(gt_image)]).to(args.device)
        
        char_metrics.update(pred_tensor, gt_tensor)
        scores = char_metrics.compute()
        
        # New Structural Metrics
        with torch.no_grad():
            skel_l1, skel_iou, edge_l1 = calculate_structural_metrics(pred_tensor, gt_tensor)
            
        results[char] = {
            "SSIM": scores["ssim"],
            "L1_Pixel": scores["l1"],
            "L1_Skeleton": skel_l1,
            "IoU_Skeleton": skel_iou,
            "L1_Edge": edge_l1
        }
        
        print(f"  SSIM: {scores['ssim']:.4f} | Skel L1(↓): {skel_l1:.4f} | Skel IoU(↑): {skel_iou:.4f} | Edge L1(↓): {edge_l1:.4f}")

    print("\n=== ZERO-SHOT SUMMARY REPORT ===")
    print(f"{'Char':<6} | {'SSIM (↑)':<10} | {'L1 Pix(↓)':<10} | {'Skel L1(↓)':<12} | {'Skel IoU(↑)':<12} | {'Edge L1(↓)':<12}")
    print("-" * 75)
    
    avg_ssim, avg_l1, avg_skel, avg_iou, avg_edge = 0, 0, 0, 0, 0
    count = len(results)
    
    for char, mets in results.items():
        print(f"{char:<6} | {mets['SSIM']:<10.4f} | {mets['L1_Pixel']:<10.4f} | {mets['L1_Skeleton']:<12.4f} | {mets['IoU_Skeleton']:<12.4f} | {mets['L1_Edge']:<12.4f}")
        avg_ssim += mets['SSIM']
        avg_l1 += mets['L1_Pixel']
        avg_skel += mets['L1_Skeleton']
        avg_iou += mets['IoU_Skeleton']
        avg_edge += mets['L1_Edge']
        
    if count > 0:
        print("-" * 75)
        print(f"{'AVG':<6} | {avg_ssim/count:<10.4f} | {avg_l1/count:<10.4f} | {avg_skel/count:<12.4f} | {avg_iou/count:<12.4f} | {avg_edge/count:<12.4f}")

if __name__ == "__main__":
    main()