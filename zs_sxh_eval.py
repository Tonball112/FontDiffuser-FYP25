# Zero-Shot Specialized Evaluation Script

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image

from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics

def extract_skeleton(img_tensor):
    dilated = F.max_pool2d(img_tensor, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-img_tensor, kernel_size=3, stride=1, padding=1)
    return dilated - eroded

def main():
    args = arg_parse()
    
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        raise ValueError("You must provide --ckpt_dir in the command line.")
    
    # --- CONFIGURATION ---
    content_dir = "data_sxh/train/ContentImage"
    
    # This is the folder YOU MUST CREATE containing the 4 real target images you deleted
    ground_truth_dir = "data_sxh/zero_shot_ground_truth" 
    
    # The reference style we are using to prompt the model
    style_image_path = "data_sxh/train/TargetImage/sxh/sxh+道.png" 
    
    output_dir = "outputs/eval_results/zero_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    # The 4 holdout characters
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
    args.seed = 1234 # Fixed seed for reproducible generation
    
    pipe = load_fontdiffuser_pipeline(args=args)
    toTensor = TF.ToTensor()
    style_image = Image.open(style_image_path).convert("RGB")

    print("\n=== STARTING ZERO-SHOT EVALUATION ===")
    
    results = {}
    
    for char in test_chars:
        print(f"\nEvaluating: {char}")
        
        # 1. Load Content (Skeleton)
        content_img = Image.open(f"{content_dir}/{char}.png").convert("RGB")
        
        # 2. Generate Zero-Shot Image
        sampling_args = dict(args=args, pipe=pipe, content_image=content_img, style_image=style_image)
        out_image = sampling(**sampling_args)
        out_image.save(f"{output_dir}/generated_{char}.png")
        
        # 3. Load Ground Truth
        gt_path = f"{ground_truth_dir}/{char}.png"
        if not os.path.exists(gt_path):
            print(f"[ERROR] Ground truth not found at {gt_path}. Cannot compute metrics for {char}!")
            continue
            
        gt_image = Image.open(gt_path).convert("RGB")
        
        # Ensure 96x96 matching
        if gt_image.size != (96, 96):
            gt_image = gt_image.resize((96, 96), Image.Resampling.BILINEAR)
            
        # 4. Compute Metrics for this specific character
        char_metrics = FontMetrics(device=args.device)
        
        pred_tensor = torch.stack([toTensor(out_image)]).to(args.device)
        gt_tensor = torch.stack([toTensor(gt_image)]).to(args.device)
        
        char_metrics.update(pred_tensor, gt_tensor)
        scores = char_metrics.compute()
        
        # Compute Skeleton L1
        with torch.no_grad():
            pred_skel = extract_skeleton(pred_tensor)
            targ_skel = extract_skeleton(gt_tensor)
            skel_l1 = F.l1_loss(pred_skel, targ_skel).item()
            
        results[char] = {
            "SSIM": scores["ssim"],
            "LPIPS": scores["lpips"],
            "L1_Pixel": scores["l1"],
            "L1_Skeleton": skel_l1
        }
        
        print(f"  SSIM: {scores['ssim']:.4f} | LPIPS: {scores['lpips']:.4f} | L1(Pix): {scores['l1']:.4f} | L1(Skel): {skel_l1:.4f}")

    # Print Final Summary Table
    print("\n=== ZERO-SHOT SUMMARY REPORT ===")
    print(f"{'Character':<10} | {'SSIM (↑)':<10} | {'LPIPS (↓)':<10} | {'L1 Pixel (↓)':<12} | {'L1 Skeleton (↓)':<12}")
    print("-" * 65)
    
    avg_ssim, avg_lpips, avg_l1, avg_skel = 0, 0, 0, 0
    count = len(results)
    
    for char, mets in results.items():
        print(f"{char:<10} | {mets['SSIM']:<10.4f} | {mets['LPIPS']:<10.4f} | {mets['L1_Pixel']:<12.4f} | {mets['L1_Skeleton']:<12.4f}")
        avg_ssim += mets['SSIM']
        avg_lpips += mets['LPIPS']
        avg_l1 += mets['L1_Pixel']
        avg_skel += mets['L1_Skeleton']
        
    if count > 0:
        print("-" * 65)
        print(f"{'AVERAGE':<10} | {avg_ssim/count:<10.4f} | {avg_lpips/count:<10.4f} | {avg_l1/count:<12.4f} | {avg_skel/count:<12.4f}")

if __name__ == "__main__":
    main()