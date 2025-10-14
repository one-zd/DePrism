import argparse
import json
import subprocess
import os
from canny_fusion import load_mask_from_file, construct_combined_sketch
from scripts.controlnet import run_controlnet
import cv2
from final_image_fusion import run_privacy_fusion
import gc
import torch
import datetime
import subprocess
import time
import tracemalloc
import torch
import os
import psutil
import functools


def release_resources():
    """释放 CPU 和 GPU 内存资源。"""
    torch.cuda.empty_cache()
    gc.collect()


def run_sam_with_profile(cmd):
    """
    封装子进程调用，统一统计：
    1. Wall 时间
    2. 进程树 CPU 内存峰值（含子进程）
    3. GPU 显存峰值（含子进程 CUDA 分配）
    """
    # 1. 初始化采样
    tracemalloc.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    parent = psutil.Process()               # 当前进程
    parent.cpu_percent()                    # 第一次采样 = 0

    # 2. 启动子进程
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cpu_peak = 0
    while proc.poll() is None:
        # 统计当前进程 + 所有子进程 RSS
        rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
        cpu_peak = max(cpu_peak, rss)
        time.sleep(0.05)                    # 100 ms 采样一次
    outs, errs = proc.communicate()

    elapsed = time.perf_counter() - t0

    # 3. GPU 峰值
    torch.cuda.synchronize()
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**3   # GB

    # 4. 清理 & 输出
    tracemalloc.stop()
    print("=== SAM 子进程性能 ===")
    print(f"返回码        : {proc.returncode}")
    print(f"Wall 时间     : {elapsed:.3f} s")
    print(f"CPU 内存峰值  : {cpu_peak / 1024**3:.2f} GB")
    print(f"GPU 显存峰值  : {gpu_peak:.2f} GB")
    if proc.returncode != 0:
        print("STDERR:", errs.decode()[-500:])
    return proc.returncode == 0



def profile_ctl(func):
    """
    装饰器：统计 ControlNet 推理
    1. Wall 时间
    2. 进程树 CPU 内存峰值
    3. GPU 显存峰值
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        proc = psutil.Process()
        proc.cpu_percent()                       # 初始化
        tracemalloc.start()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            out = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            torch.cuda.synchronize()
            # 整棵树 RSS
            rss = sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)])
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3   # GB
            tracemalloc.stop()

        print("=== ControlNet 性能 ===")
        print(f"Wall 时间     : {elapsed:.3f} s")
        print(f"CPU 内存峰值  : {rss / 1024**3:.2f} GB")
        print(f"GPU 显存峰值  : {gpu_peak:.2f} GB")
        return out
    return wrapper






def run_sam_cmd(
    source_image_path: str,
    output_dir: str,
    classes: str,
    sam_script_path: str = "E:\phd//3\code\Grounded-Segment-Anything-main\EfficientSAM\pp_sam_cmd.py",  # 你的 SAM 脚本路径
    conda_env: str = "GSA",
    device: str = "cuda",
    grounding_dino_config_path: str = "E:/phd/3/code/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint_path: str = "E:/phd/3/code/Grounded-Segment-Anything-main/models/grounding-dino/groundingdino_swint_ogc.pth",
    repvit_sam_checkpoint_path: str = "E:/phd/3/code/Grounded-Segment-Anything-main/EfficientSAM/RepViTSAM/repvit_sam.pt",
    box_threshold: float = 0.45,
    text_threshold: float = 0.2,
    nms_threshold: float = 0.8):
    """
    使用 subprocess 以 conda 虚拟环境运行 SAM 模块，并传递参数。
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "conda", "run", "-n", conda_env, "python", sam_script_path,
        "--source_image_path", source_image_path,
        "--output_dir", output_dir,
        "--classes", classes,
        "--device", device,
        "--grounding_dino_config_path", grounding_dino_config_path,
        "--grounding_dino_checkpoint_path", grounding_dino_checkpoint_path,
        "--repvit_sam_checkpoint_path", repvit_sam_checkpoint_path,
        "--box_threshold", str(box_threshold),
        "--text_threshold", str(text_threshold),
        "--nms_threshold", str(nms_threshold),
    ]

    print("[run_sam_cmd] Launching:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print("[run_sam_cmd] Error occurred:\n", result.stderr)
        raise RuntimeError("SAM command failed.")
    else:
        print("[run_sam_cmd] Completed successfully.")
        print(result.stdout)

    #####开销计算####
    # success = run_sam_with_profile(cmd)




def run_llm(
    prompt,
    sensitive_words,
    json_output,
    model="deepseek-r1:32b",
    temperature_list=[1.5],
    api_url="http://localhost:3000/ollama/api/chat",
    max_examples=None
):
    """
    调用 LLM 脚本进行隐私重构处理，支持所有重要参数。
    """

    # 构建温度参数：支持单个或多个值
    temp_args = " ".join([str(t) for t in temperature_list])

    # 构建命令字符串
    cmd = (
        f'conda run -n control python E:/phd/3/code/ppi_control/scripts/llm_text_sec.py '
        f'--prompt "{prompt}" '
        f'--sensitive_words "{sensitive_words}" '
        f'--output_dir "{json_output}" '
        f'--model "{model}" '
        f'--temperature {temp_args} '
        f'--api_url "{api_url}"'
    )

    if max_examples is not None:
        cmd += f' --max_examples {max_examples}'

    print(f"[LLM] Running command:\n{cmd}")


    result = subprocess.run(cmd, shell=True, check=True)

    if result.returncode != 0:
        raise RuntimeError("LLM 脚本执行失败")

    if not os.path.exists(json_output):
        raise FileNotFoundError(f"LLM 输出未生成：{json_output}")



def parse_llm_result(output_dir):
    base_name = f"llm_text_sec_deepseek_32b_1.5"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "r") as f:
        data = json.load(f)[0]
    camouflage_infos = []
    for word, val in data["perturbed_sentence"]["camouflage"].items():
        camouflage_infos.append({
            "word": word,
            "prompt": val["camouflage prompt"],
            "elements": val["camouflage element"]
        })
    return data["original_sentence"], data["perturbed_sentence"]["reduced_sentence"], camouflage_infos


def run_camouflage_element_cmd(
    output_dir: str,
    categorys: list,
    count: int = 5,
    img_size: int = 128,
    force: bool = False,
    shape_script_path: str = "E:\phd//3\code\ppi_control\scripts\generate_camouflage_canny.py",  # 你的主脚本路径
    conda_env: str = "control",  # 替换为你生成 shape 图标的 conda 虚拟环境
):
    """
    使用 subprocess 以 conda 虚拟环境运行 shape 图标生成脚本，并传递参数。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令行参数
    cmd = [
        "conda", "run", "-n", conda_env, "python", shape_script_path,
        "--output_dir", output_dir,
        "--categories", json.dumps(categorys),
        "--count", str(count),
        "--img_size", str(img_size),
    ]

    # #####开销计算####
    # success = run_sam_with_profile(cmd)

    if force:
        cmd.append("--force")

    print("[run_shape_generator_cmd] Launching:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("[run_shape_generator_cmd] Command failed with return code:", e.returncode)
        raise RuntimeError("Shape generator command execution failed.")
    except Exception as e:
        print("[run_shape_generator_cmd] Failed to start subprocess:", str(e))
        raise


def run_canny_fusion(mask_path, camouflage_elements, shape_library_root="E:/phd/3/code/ppi_control/shape_library", image_size=512):
    """
    将隐私区域 mask 与多个 camouflage 元素融合为统一风格的 Canny 草图。

    参数:
        mask_path: 隐私区域的 canny mask 路径
        camouflage_elements: 元素类别列表，例如 ["wooden chess set", "desk lamp"]
        shape_library_root: 图库路径
        image_size: 输出图像尺寸
    返回:
        融合后的 PIL.Image 草图图像
    """
    if isinstance(camouflage_elements, list):
        elements_str = ", ".join(camouflage_elements)
    elif isinstance(camouflage_elements, str):
        elements_str = camouflage_elements
    else:
        raise ValueError("camouflage_elements 必须是字符串或字符串列表")

    # 加载 base mask
    base_mask = load_mask_from_file(mask_path, size=image_size)

    # 构建合成图像
    final_canny = construct_combined_sketch(
        element_names=elements_str,
        existing_mask=base_mask,
        shape_library_root=shape_library_root,
        image_size=image_size
    )
    return final_canny





def main(args):

    # 保存当前运行参数，带时间戳
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args_log_path = os.path.join(args.output_dir, f"args_logger_{timestamp}.json")

    # 把 argparse 参数转成 dict 保存
    args_dict = vars(args)
    with open(args_log_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    print(f"[Logger] 参数已保存到: {args_log_path}")

    # t0 = time.perf_counter()
    #
    # # A. SAM 模块（另一个conda环境，cmd独立调用）
    print("Running SAM module...")
    run_sam_cmd(classes=args.sensitive_words, source_image_path=args.image_path, output_dir=args.output_dir)

    # samelapsed = time.perf_counter() - t0
    # print(f"samelapsed 时间     : {samelapsed:.3f} s")


    #  解析多敏感词的 Canny 和 Mask 图
    sensitive_words = [word.strip() for word in args.sensitive_words.split(',')]
    privacy_canny_paths = [os.path.join(args.output_dir, f'{word}_canny.png') for word in sensitive_words]
    privacy_mask_paths = [os.path.join(args.output_dir, f'{word}_mask.png') for word in sensitive_words]
    background_canny_path = os.path.join(args.output_dir, f'background_canny.png')
    #
    #

    #
    # t1 = time.perf_counter()
    # # B. LLM 模块（显存不够，cmd独立调用）
    # print("Running LLM module...")
    run_llm(prompt=args.prompt, sensitive_words=args.sensitive_words, json_output=args.output_dir)
    _, reduced_sentence, camouflage_infos = parse_llm_result(output_dir=args.output_dir)

    # llm_elapsed = time.perf_counter() - t1
    # print(f"llm_elapsed时间     : {llm_elapsed:.3f} s")

    # #
    #
    #
    # # ####直接controlnet生成真值
    # t3 = time.perf_counter()
    print("Generating  GT image...")
    run_controlnet(input_image_path=args.image_path, prompt=args.prompt, output_path=args.output_dir,output_name='gt', a_prompt=args.a_prompt,n_prompt=args.n_prompt,method=args.method,
                       scale=args.scale, ddim_steps=args.ddim_steps, strength=args.strength, seed=args.seed,low_threshold=args.low_threshold, high_threshold=args.high_threshold)
    release_resources()
    #
    # controlnet_elapsed = time.perf_counter() - t3
    # print(f"controlnet_elapsed时间     : {controlnet_elapsed:.3f} s")
    #
    # C. 用 reduced_sentence + other_canny 生成背景图
    print("Generating background image...")
    run_controlnet(input_image_path=background_canny_path, prompt=reduced_sentence,output_path=args.output_dir, output_name='background',a_prompt=args.a_prompt,n_prompt=args.n_prompt,method=args.method,
                       scale=args.scale, ddim_steps=args.ddim_steps, strength=args.strength, seed=args.seed, low_threshold=args.low_threshold, high_threshold=args.high_threshold)
    release_resources()
    controlnet_background_path = os.path.join(args.output_dir, f'controlnet_background.png')
    #
    #
    #
    # D-E. 每个敏感词分别处理
    camouflage_generated_list = []
    for idx, camo in enumerate(camouflage_infos):
        privacy_word = camo["word"]
        camouflage_prompt = camo["prompt"]
        # camouflage_elements = camo["elements"]
        camouflage_elements = camo["elements"][:0]    #######限定扰动元素数量
        # camouflage_elements_name = "_".join(elem.replace(" ", "_") for elem in camouflage_elements)

        print(f"Processing sensitive word: {privacy_word}")
        print(f"Camouflage prompt: {camouflage_prompt}")

        # E1. 生成伪装元素 Canny
        # t2 = time.perf_counter()
        run_camouflage_element_cmd(output_dir="E:\\phd\\3\\code\\ppi_control\\shape_library",
                                   categorys=camouflage_elements, count=1, img_size=128, force=False)
        release_resources()
        # camouflage_elapsed = time.perf_counter() - t2
        # print(f"camouflage_elapsed时间     : {camouflage_elapsed:.3f} s")


        # E2. 融合当前敏感词对应的 Canny
        fused_canny = run_canny_fusion(privacy_canny_paths[idx], camouflage_elements)


        fused_canny_path = os.path.join(args.output_dir, f'canny_{privacy_word}.png')
        fused_canny.save(fused_canny_path)

        # E3. ControlNet 根据融合的 canny+prompt 生成伪装图
        run_controlnet(input_image_path=fused_canny_path,
                       prompt=camouflage_prompt, output_path=args.output_dir,
                       output_name=f'camouflage_{privacy_word}',
                       a_prompt=args.a_prompt, n_prompt=args.n_prompt,method=args.method,
                       scale=args.scale, ddim_steps=args.ddim_steps, strength=args.strength, seed=args.seed,
                       low_threshold=args.low_threshold, high_threshold=args.high_threshold)
        release_resources()

        controlnet_camouflage_path = os.path.join(args.output_dir, f'controlnet_camouflage_{privacy_word}.png')
        camouflage_image = cv2.imread(controlnet_camouflage_path)

        camouflage_generated_list.append({
            "image": camouflage_image,
            "mask": privacy_mask_paths[idx]
        })
    #
    # #
    # # F. 最后融合所有敏感区域
    print("Fusing all camouflage results into final output...")

    final_img = cv2.imread(controlnet_background_path)
    step_idx = [0]  # 用列表传引用，保证 run_privacy_fusion 内可以自增

    for item in camouflage_generated_list:
        final_img = run_privacy_fusion(
            base_image=final_img,
            camouflage_images=item["image"],
            mask_image=item["mask"],
            output_dir=args.output_dir,
            intermediate=True,
            step_idx=step_idx  # 传步数计数器
        )
    #
    # # G. 保存最终融合结果
    cv2.imwrite(os.path.join(args.output_dir, "final_fused_img.png"), final_img)
    print(f"✅ Done. Final result saved to: {args.output_dir}")

    release_resources()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full privacy-preserving generation pipeline.")
    parser.add_argument("--image_path", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100\other\image_484.png', help="Path to input image")
    parser.add_argument("--prompt", type=str, default='A bowl of vegetable curry with broccoli and carrots, served on a wooden table, accompanied by bowls of turmeric powder and another light colored spice powder.', help="Full input prompt")
    parser.add_argument("--sensitive_words",type=str, default='A bowl of vegetable curry,wooden table', help="Comma-separated list of sensitive words")
    parser.add_argument("--output_dir", type=str, default="E:\phd//3\code\ppi_control\outputs//image_484//", help="Path to save final result")
    parser.add_argument("--a_prompt", type=str,
                        default=" Qi Baishi style,best quality,masterpiece,Simple background",
                        help="风格、积极提示词,Qi Baishi style,cartoon style,Chinese style interior,Achinese ink painting, classic Japanese anime art style,delicate face，")
    parser.add_argument("--n_prompt", type=str,
                        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                        help="消极提示词")
    parser.add_argument('--method', type=str, default='controlnet',
                        choices=['controlnet', 't2i_adapter','unicontrolnet','controlar'],
                        help='Choose the control t2i method')
    #ControlNet 相关参数
    parser.add_argument("--scale", type=float, default=10, help="Guidance scale for ControlNet")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps for ControlNet")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--low_threshold", type=int, default=100, help="Canny edge low threshold")
    parser.add_argument("--high_threshold", type=int, default=200, help="Canny edge high threshold")


    args = parser.parse_args()
    main(args)
