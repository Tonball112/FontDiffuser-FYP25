# Modified for SXH Dataset Evaluation (Term 2 CASD Architecture)
# Includes Advanced Structural Metrics (Skeleton IoU & Sobel Edge)

import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import yaml
from PIL import Image

from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics

def load_essential_args(args, guidance_scale: float = 7.5):
    args.guidance_type = "classifier-free"
    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    args.guidance_scale = guidance_scale
    return args

def run_fontdiffuser_demo_mode(
    args, pipe, content_image: Optional[Image.Image], character: Optional[str],
    style_images: list[Image.Image], ttf_path: str, use_few_shot: bool,
    num_inference_steps: int = 20, batch_size: int = 1, seed: Optional[int] = None
):
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"
    args.demo = True
    args.character_input = False if content_image is not None else True
    args.content_character = character
    args.num_inference_steps = num_inference_steps
    args.ttf_path = ttf_path
    args.batch_size = batch_size
    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    sampling_args = dict[str, Any](
        args=args, pipe=pipe, content_image=content_image,
    )
    if use_few_shot:
        sampling_args["style_images"] = style_images
    else:
        sampling_args["style_image"] = style_images[0]

    return sampling(**sampling_args)

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

def generate_single_test(num_style_image: int, dataset_files: list[Path]):
    test_info = {}
    for file_idx in range(len(dataset_files)):
        available_style_choices = list(range(len(dataset_files)))
        chosen_styles: list[int] = []

        if file_idx in available_style_choices:
            available_style_choices.remove(file_idx)

        while len(chosen_styles) < num_style_image:
            if not available_style_choices: break 
            style = random.choice(available_style_choices)
            chosen_styles.append(style)
            available_style_choices.remove(style)

        styles = [dataset_files[i] for i in chosen_styles]
        test_info[file_idx] = {
            "character": dataset_files[file_idx].name,
            "style": [file.name for file in styles],
        }
    return test_info

def create_test_profile(profile_dir: str, num_test_round: int, num_style_image: int, dataset_files: list[Path]):
    os.makedirs(profile_dir, exist_ok=True)
    for test_idx in range(num_test_round):
        seed = random.randint(0, 10000)
        test_info = generate_single_test(num_style_image=num_style_image, dataset_files=dataset_files)
        test_configuration = {"index": test_idx, "test_info": test_info, "seed": seed}
        with open(f"{profile_dir}/test_{test_idx}.yaml", "w", encoding="utf-8") as yaml_file:
            yaml.dump(test_configuration, yaml_file, default_flow_style=False, allow_unicode=True)
    print(f"[Eval] Test profile created at {profile_dir} ({num_test_round} tests)")

def load_test_profile(profile_dir: str):
    if not os.path.exists(profile_dir): return None
    profile_files = [f for f in Path(profile_dir).iterdir() if f.suffix == '.yaml']
    if len(profile_files) == 0: return None

    profile_info_map = {}
    for profile_file in profile_files:
        with open(profile_file, "r", encoding="utf-8") as yaml_file:
            test_configuration = yaml.load(yaml_file, Loader=yaml.FullLoader)
            profile_info_map[test_configuration["index"]] = test_configuration

    profile_info = []
    test_idx = 0
    while test_idx in profile_info_map:
        profile_info.append(profile_info_map[test_idx])
        test_idx += 1
    return profile_info

def save_results(result_info: dict, output_dir: str):
    with open(f"{output_dir}/eval_results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(result_info, yaml_file, default_flow_style=False, allow_unicode=True)

def parse_target_image_name(target_image_name: str):
    target_components = target_image_name.split("+")
    style = target_components[0]
    content = target_components[1]
    return style, content

def main():
    args = arg_parse()
    
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        raise ValueError("You must provide --ckpt_dir in the command line.")

    dataset_dir = "data_sxh/train/TargetImage/sxh"
    content_dir = "data_sxh/train/ContentImage"
    ttf_path = "ttf/KaiXinSongA.ttf" 

    use_few_shot = False 
    num_test_round = 1     
    num_style_image = 5 
    expect_existing_profile = False 

    ckpt_name = Path(args.ckpt_dir).name
    test_profile_dir = "outputs/eval_profiles/sxh_profile"
    results_output_dir = f"outputs/eval_results/sxh_{ckpt_name}_eval"

    dataset_dir_path = Path(dataset_dir)
    dataset_files = [f for f in dataset_dir_path.iterdir() if f.suffix in ['.png', '.jpg']]

    profile_info = load_test_profile(profile_dir=test_profile_dir)
    if profile_info is None:
        if expect_existing_profile: raise ValueError(f"Test profile expected but not found.")
        create_test_profile(test_profile_dir, num_test_round, num_style_image, dataset_files)
        profile_info = load_test_profile(profile_dir=test_profile_dir)

    load_essential_args(args=args)
    pipe = load_fontdiffuser_pipeline(args=args)
    toTensor = TF.ToTensor()

    overall_performance = FontMetrics(device=args.device)
    test_results = {}
    
    all_skel_l1, all_skel_iou, all_edge_l1 = [], [], []

    os.makedirs(results_output_dir, exist_ok=True)
    save_results(test_results, results_output_dir)

    total_tests = len(profile_info)

    for test_idx, test_info in enumerate(profile_info):
        seed = test_info["seed"]
        test_performance = FontMetrics(device=args.device)
        round_skel_l1, round_skel_iou, round_edge_l1 = [], [], []

        os.makedirs(f"{results_output_dir}/{test_idx}", exist_ok=True)
        total_files = len(test_info["test_info"])

        for file_idx, file_info in enumerate(test_info["test_info"].values()):
            print(f"[{test_idx + 1}/{total_tests}][{file_idx + 1}/{total_files}] ", end="", flush=True)

            character_file = Path(f"{dataset_dir}/{file_info['character']}")
            character_image = Image.open(character_file).convert("RGB")
            
            style_files = [Path(f"{dataset_dir}/{style}") for style in file_info["style"]]
            style_images = [Image.open(f).convert("RGB") for f in style_files]

            _, character_char = parse_target_image_name(character_file.stem)

            content_image_path = Path(f"{content_dir}/{character_char}.png")
            if not content_image_path.exists():
                content_image_path = Path(f"{content_dir}/{character_char}.jpg")
            if not content_image_path.exists(): continue

            content_image = Image.open(content_image_path).convert("RGB")

            out_image = run_fontdiffuser_demo_mode(
                args=args, pipe=pipe, content_image=content_image, 
                character=character_char, style_images=style_images,
                ttf_path=ttf_path, use_few_shot=use_few_shot, seed=seed,
            )
            out_image.save(f"{results_output_dir}/{test_idx}/{character_char}.png")

            if character_image.size != (96, 96):
                character_image = character_image.resize((96, 96), Image.Resampling.BILINEAR)

            output_image_batch = torch.stack([toTensor(out_image)]).to(args.device)
            character_image_batch = torch.stack([toTensor(character_image)]).to(args.device)

            test_performance.update(output_image_batch, character_image_batch)
            overall_performance.update(output_image_batch, character_image_batch)

            # Compute new metrics
            skel_l1, skel_iou, edge_l1 = calculate_structural_metrics(output_image_batch, character_image_batch)
            round_skel_l1.append(skel_l1)
            round_skel_iou.append(skel_iou)
            round_edge_l1.append(edge_l1)
            
            all_skel_l1.append(skel_l1)
            all_skel_iou.append(skel_iou)
            all_edge_l1.append(edge_l1)

        test_performance_result = test_performance.compute()
        test_performance_result["skeleton_l1"] = sum(round_skel_l1) / len(round_skel_l1) if round_skel_l1 else 0
        test_performance_result["skeleton_iou"] = sum(round_skel_iou) / len(round_skel_iou) if round_skel_iou else 0
        test_performance_result["edge_l1"] = sum(round_edge_l1) / len(round_edge_l1) if round_edge_l1 else 0
        
        test_results[test_idx] = test_performance_result
        save_results(test_results, results_output_dir)

    overall_performance_result = overall_performance.compute()
    overall_performance_result["skeleton_l1"] = sum(all_skel_l1) / len(all_skel_l1) if all_skel_l1 else 0
    overall_performance_result["skeleton_iou"] = sum(all_skel_iou) / len(all_skel_iou) if all_skel_iou else 0
    overall_performance_result["edge_l1"] = sum(all_edge_l1) / len(all_edge_l1) if all_edge_l1 else 0

    test_results["mean"] = {
        "fid": sum([res["fid"] for res in test_results.values() if "fid" in res]) / max(1, len(test_results) - 1),
        "ssim": sum([res["ssim"] for res in test_results.values() if "ssim" in res]) / max(1, len(test_results) - 1),
        "lpips": sum([res["lpips"] for res in test_results.values() if "lpips" in res]) / max(1, len(test_results) - 1),
        "l1": sum([res["l1"] for res in test_results.values() if "l1" in res]) / max(1, len(test_results) - 1),
        "skeleton_l1": overall_performance_result["skeleton_l1"],
        "skeleton_iou": overall_performance_result["skeleton_iou"],
        "edge_l1": overall_performance_result["edge_l1"]
    }

    test_results["overall_performance"] = overall_performance_result
    save_results(test_results, results_output_dir)

    print("\n[Eval] Evaluation finished")
    print("Overall performance: \n"
          f"\tFID:         {overall_performance_result['fid']:.4f}\n"
          f"\tSSIM:        {overall_performance_result['ssim']:.4f}\n"
          f"\tLPIPS:       {overall_performance_result['lpips']:.4f}\n"
          f"\tL1 (Pixel):  {overall_performance_result['l1']:.4f}\n"
          f"\tL1 (Skeleton): {overall_performance_result['skeleton_l1']:.4f}\n"
          f"\tIoU (Skeleton): {overall_performance_result['skeleton_iou']:.4f}\n"
          f"\tL1 (Edge):   {overall_performance_result['edge_l1']:.4f}")

if __name__ == "__main__":
    main()