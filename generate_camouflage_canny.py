import os
import argparse
import json
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionPipeline
from rembg import remove
import numpy as np
from controlnet_aux.processor import CannyDetector
from collections import defaultdict
device = "cuda" if torch.cuda.is_available() else "cpu"



def load_index(index_path):
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index(index, index_path):
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def generate_blank_scribble(size=128):
    return Image.new("RGB", (size, size), color="white")

def generate_prompt(category):
    return [f"High quality,exquisite,a simple line draw of a {category}, isolated"]

def remove_background(img: Image.Image) -> Image.Image:
    return remove(img)  # 返回 RGBA 图像

def convert_to_canny(pil_img: Image.Image, out_size: int, low_threshold=100, high_threshold=200) -> Image.Image:
    np_img = np.array(pil_img.convert("RGB"))
    detector = CannyDetector()
    canny_np = detector(np_img, low_threshold=low_threshold, high_threshold=high_threshold)
    canny_img = Image.fromarray(canny_np)
    return canny_img.resize((out_size, out_size), Image.LANCZOS)

def main(args):


    # 加载全局索引（改为类别 → 图像列表）
    index_path = os.path.join(args.output_dir, "index.json")
    index = load_index(index_path) or {}
    index = defaultdict(list, index)  # 支持 index["类别"] 直接 append


    # 初始化模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "E:\phd//3\code\ppi_control\stable-diffusion-v1-5stable-diffusion-v1-5",  # 替换为你本地模型路径
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    categories = json.loads(args.categories)
    for category in categories:
        print(f"\n⏳ 开始生成类别：{category}")
        output_dir = os.path.join(args.output_dir, category)
        os.makedirs(output_dir, exist_ok=True)

        existing_files = index.get(category, [])
        if existing_files and not args.force:
            print(f"已存在 {len(existing_files)} 张 '{category}' 图像，跳过生成。使用 --force 可覆盖。")
            continue

        prompts = generate_prompt(category)
        idx_start = len(existing_files) + 1

        for i in range(args.count):
            prompt_text = prompts[i % len(prompts)]
            control_img = generate_blank_scribble(args.img_size)

            result = pipe(prompt=prompt_text, height=512, width=512).images[0]

            cleaned = remove_background(result)
            base_name = f"{category}_{idx_start + i:03d}"
            canny_filename = f"{base_name}_canny.png"
            canny_path = os.path.join(output_dir, canny_filename)

            canny_img = convert_to_canny(cleaned, args.img_size)
            canny_img.save(canny_path)

            index[category].append(canny_path)
            print(f"✔ 已保存：{canny_path}")

    # 保存索引（转为普通 dict 以便 json 序列化）
    save_index(dict(index), index_path)
    print(f"索引已更新：{index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量生成多个类别的线稿图像")
    parser.add_argument("--output_dir", type=str, default="E:\\phd\\3\\code\\ppi_control\\shape_library", help="输出目录")
    # parser.add_argument("--categories", nargs="+", type=str, default=["desk lamp"], help="类别列表，例如 book vase")
    parser.add_argument(
        "--categories", type=str,default='[\"dog\", \"duck\"]', help="以 JSON 列表格式传入类别，例如 '[\"wooden chess set\", \"desk lamp\"]'"
    )

    parser.add_argument("--count", type=int, default=5, help="每个类别生成的图像数量")
    parser.add_argument("--img_size", type=int, default=128, help="输出图像尺寸（正方形）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有图库")

    args = parser.parse_args()
    main(args)

