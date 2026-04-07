# Modified for SXH Dataset Evaluation
# Based on lantingjixu_eval.py provided by FYP24

import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms as TF
import yaml
from PIL import Image

# Import necessary modules from your existing project structure
from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics


def load_essential_args(
    args,
    ckpt_dir: str,
    guidance_scale: float = 7.5,
):
    args.guidance_type = "classifier-free"
    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    args.ckpt_dir = ckpt_dir
    args.guidance_scale = guidance_scale
    return args


def run_fontdiffuser_demo_mode(
    args,
    pipe,
    content_image: Optional[Image.Image],
    character: Optional[str],
    style_images: list[Image.Image],
    ttf_path: str,
    use_few_shot: bool,
    num_inference_steps: int = 20,
    batch_size: int = 1,
    seed: Optional[int] = None,
):
    # You can change this to "pmndm" if you want to match the default training sampler
    # But dpmsolver++ is generally faster for evaluation
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"

    args.demo = True

    # Crucial Logic: If content_image is provided, use it. Otherwise use character string.
    args.character_input = False if content_image is not None else True
    args.content_character = character
    args.num_inference_steps = num_inference_steps
    args.ttf_path = ttf_path
    args.batch_size = batch_size

    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    sampling_args = dict[str, Any](
        args=args,
        pipe=pipe,
        content_image=content_image,
    )
    if use_few_shot:
        sampling_args["style_images"] = style_images
    else:
        # For single shot, we just take the first style image provided
        sampling_args["style_image"] = style_images[0]

    out_image = sampling(**sampling_args)
    return out_image


def generate_single_test(num_style_image: int, dataset_files: list[Path]):
    test_info = {}
    for file_idx in range(len(dataset_files)):
        available_style_choices = list(range(len(dataset_files)))
        chosen_styles: list[int] = []

        # Ensure we don't use the Ground Truth image as the Style Input (that would be cheating)
        if file_idx in available_style_choices:
            available_style_choices.remove(file_idx)

        # Pick random style images from the rest of the dataset
        while len(chosen_styles) < num_style_image:
            if not available_style_choices:
                break 
            style = random.choice(available_style_choices)
            chosen_styles.append(style)
            available_style_choices.remove(style)

        styles = [dataset_files[i] for i in chosen_styles]

        test_info[file_idx] = {
            "character": dataset_files[file_idx].name,
            "style": [file.name for file in styles],
        }
    return test_info


def create_test_profile(
    profile_dir: str,
    num_test_round: int,
    num_style_image: int,
    dataset_files: list[Path],
):
    os.makedirs(profile_dir, exist_ok=True)
    for test_idx in range(num_test_round):
        seed = random.randint(0, 10000)
        test_info = generate_single_test(
            num_style_image=num_style_image, dataset_files=dataset_files
        )
        test_configuration = {
            "index": test_idx,
            "test_info": test_info,
            "seed": seed,
        }
        with open(f"{profile_dir}/test_{test_idx}.yaml", "w", encoding="utf-8") as yaml_file:
            yaml.dump(test_configuration, yaml_file, default_flow_style=False, allow_unicode=True)
    print(f"[Eval] Test profile created at {profile_dir} ({num_test_round} tests)")


def load_test_profile(profile_dir: str):
    if not os.path.exists(profile_dir):
        return None
    profile_files = [f for f in Path(profile_dir).iterdir() if f.suffix == '.yaml']
    if len(profile_files) == 0:
        return None

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
    # Input Format: sxh+Character.png
    # Splits by '+'
    target_components = target_image_name.split("+")
    style = target_components[0]
    content = target_components[1]
    return style, content


def main():
    args = arg_parse()
    
    # --- CONFIGURATION START ---
    
    # 1. Checkpoint Directory (Your 180k step model)
    ckpt_dir = "outputs/FontDiffuser/global_step_180000"
    
    # 2. Dataset Directories
    # Where your Target (Ground Truth) images are
    dataset_dir = "data_sxh/train/TargetImage/sxh"
    # Where your Content (Skeleton) images are
    content_dir = "data_sxh/train/ContentImage"
    
    # 3. TTF Path 
    # (Just a placeholder since we use content images, but the script might check for existence)
    ttf_path = "ttf/KaiXinSongA.ttf" 

    # 4. Evaluation Settings
    use_few_shot = False # You used single shot in your manual test
    
    # Output locations
    test_profile_dir = "outputs/eval_profiles/sxh_profile"
    results_output_dir = "outputs/eval_results/sxh_180k_eval"

    # Profile Settings
    # num_test_round = 10  # Full evaluation usually does 10 rounds
    num_test_round = 1     # Setting to 1 for a quick check. Increase this for full paper results!
    
    # If use_few_shot is False, only the first image in the style list is used.
    # But we still select 5 candidates in the profile to be safe.
    num_style_image = 5 
    
    expect_existing_profile = False # Set to False to create a new profile for SXH

    # --- CONFIGURATION END ---

    ### Part 1: Load/Generate the test profile ###
    dataset_dir_path = Path(dataset_dir)
    # Filter for png/jpg to avoid reading hidden system files
    dataset_files = [f for f in dataset_dir_path.iterdir() if f.suffix in ['.png', '.jpg']]

    profile_info = load_test_profile(profile_dir=test_profile_dir)
    print()
    if profile_info is None:
        if expect_existing_profile:
            raise ValueError(f"Test profile expected at {test_profile_dir} but not found.")

        print(f"[Eval] No test profile found. Creating a new test profile...")
        create_test_profile(
            profile_dir=test_profile_dir,
            num_test_round=num_test_round,
            num_style_image=num_style_image,
            dataset_files=dataset_files,
        )
        profile_info = load_test_profile(profile_dir=test_profile_dir)
        assert profile_info is not None
        print(f"[Eval] Test profile loaded from {test_profile_dir} ({len(profile_info)} tests)")
    else:
        print(f"[Eval] Test profile loaded from {test_profile_dir} ({len(profile_info)} tests)")
    print()

    ### Part 2: Run the evaluation process ###
    print("[Eval] Evaluation begins")
    print()

    load_essential_args(args=args, ckpt_dir=ckpt_dir)
    pipe = load_fontdiffuser_pipeline(args=args)
    toTensor = TF.ToTensor()

    total_time = 0
    total_rounds = 0
    overall_performance = FontMetrics(device=args.device)
    test_results = {}

    os.makedirs(results_output_dir, exist_ok=True)
    save_results(test_results, results_output_dir)

    total_tests = len(profile_info)

    for test_idx, test_info in enumerate(profile_info):
        start_time = time.time()
        print()
        print(f"[Eval] {total_tests - test_idx} tests remaining")
        print(f"[Eval] Test {test_idx} begins (Saving to {results_output_dir}/{test_idx})")
        print()

        seed = test_info["seed"]
        test_performance = FontMetrics(device=args.device)

        os.makedirs(f"{results_output_dir}/{test_idx}", exist_ok=True)
        total_files = len(test_info["test_info"])

        for file_idx, file_info in enumerate(test_info["test_info"].values()):
            print(f"[{test_idx + 1}/{total_tests}][{file_idx + 1}/{total_files}] ", end="", flush=True)

            # 1. Load Target (Ground Truth) Image
            character_file = Path(f"{dataset_dir}/{file_info['character']}")
            character_image = Image.open(character_file).convert("RGB")
            
            # 2. Load Style (Reference) Images
            style_files = [Path(f"{dataset_dir}/{style}") for style in file_info["style"]]
            style_images = [Image.open(f).convert("RGB") for f in style_files]

            # 3. Parse Filename to get Character (e.g. "丙" from "sxh+丙")
            _, character_char = parse_target_image_name(character_file.stem)

            # 4. NEW: Load Content Image from your Content Directory
            content_image_path = Path(f"{content_dir}/{character_char}.png")
            
            if not content_image_path.exists():
                # Fallback: Try .jpg if .png doesn't exist
                content_image_path = Path(f"{content_dir}/{character_char}.jpg")
            
            if not content_image_path.exists():
                print(f"\n[Warning] Content image for {character_char} not found at {content_image_path}. Skipping.")
                continue

            content_image = Image.open(content_image_path).convert("RGB")

            # 5. Run Sampling
            out_image = run_fontdiffuser_demo_mode(
                args=args,
                pipe=pipe,
                content_image=content_image, # Pass the loaded content image
                character=character_char,
                style_images=style_images,
                ttf_path=ttf_path,
                use_few_shot=use_few_shot,
                seed=seed,
            )

            assert out_image is not None
            out_image.save(f"{results_output_dir}/{test_idx}/{character_char}.png")

            # 6. Resize if dimensions mismatch (Evaluator expects match)
            if out_image.size != character_image.size:
                out_image = out_image.resize(
                    character_image.size, Image.Resampling.BILINEAR
                )

            # 7. Compute Metrics
            output_image_batch = torch.stack([toTensor(out_image)])
            character_image_batch = torch.stack([toTensor(character_image)])

            test_performance.update(output_image_batch, character_image_batch)
            overall_performance.update(output_image_batch, character_image_batch)

        test_performance_result = test_performance.compute()
        test_results[test_idx] = test_performance_result
        save_results(test_results, results_output_dir)

        end_time = time.time()
        round_time = end_time - start_time
        total_time += round_time
        total_rounds += 1

        print(f"\n[Eval] Round {test_idx} finished in {round_time:.2f}s")

    # --- Final Aggregation ---
    overall_performance_result = overall_performance.compute()
    average_round_time = 0 if total_rounds == 0 else total_time / total_rounds

    # Calculate Mean/Std across rounds
    fid_values = torch.tensor([res["fid"] for res in test_results.values() if "fid" in res])
    ssim_values = torch.tensor([res["ssim"] for res in test_results.values() if "ssim" in res])
    lpips_values = torch.tensor([res["lpips"] for res in test_results.values() if "lpips" in res])
    l1_values = torch.tensor([res["l1"] for res in test_results.values() if "l1" in res])

    mean_result = {
        "fid": fid_values.mean().item() if len(fid_values) > 0 else 0,
        "ssim": ssim_values.mean().item() if len(ssim_values) > 0 else 0,
        "lpips": lpips_values.mean().item() if len(lpips_values) > 0 else 0,
        "l1": l1_values.mean().item() if len(l1_values) > 0 else 0,
    }

    std_result = {
        "fid": fid_values.std().item() if len(fid_values) > 0 else 0,
        "ssim": ssim_values.std().item() if len(ssim_values) > 0 else 0,
        "lpips": lpips_values.std().item() if len(lpips_values) > 0 else 0,
        "l1": l1_values.std().item() if len(l1_values) > 0 else 0,
    }

    test_results["mean"] = mean_result
    test_results["std"] = std_result
    test_results["overall_performance"] = overall_performance_result
    test_results["total_time"] = total_time

    save_results(test_results, results_output_dir)

    print()
    print("[Eval] Evaluation finished")
    print("Overall performance: \n"
          f"\tfid: {overall_performance_result['fid']:.4f}\n"
          f"\tssim: {overall_performance_result['ssim']:.4f}\n"
          f"\tlpips: {overall_performance_result['lpips']:.4f}\n"
          f"\tl1: {overall_performance_result['l1']:.4f}")

if __name__ == "__main__":
    main()
