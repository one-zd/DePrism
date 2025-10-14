import os
import numpy as np
from PIL import Image
import random
import argparse
from ControlNet.annotator.canny import CannyDetector
from scipy.ndimage import gaussian_filter


def standardize_canny(image: np.ndarray, blur_sigma=0.5):
    """
    使用统一的 Canny 检测器生成轮廓图，并做轻微模糊处理统一风格。
    """
    detector = CannyDetector()
    canny = detector(image,low_threshold = 100, high_threshold =200)
    if blur_sigma > 0:
        canny = gaussian_filter(canny, sigma=blur_sigma)
    return canny


def load_random_canny(category_folder, target_size=None):
    files = [f for f in os.listdir(category_folder) if f.endswith("_canny.png")]
    if not files:
        raise FileNotFoundError(f"No canny files found in {category_folder}")
    path = os.path.join(category_folder, random.choice(files))
    image = Image.open(path).convert("RGB")
    if target_size:
        image = image.resize((target_size, target_size), Image.LANCZOS)
    np_img = np.array(image)
    return standardize_canny(np_img)


def paste_nonoverlapping(
    base_mask,
    new_shape,
    min_scale=0.5,
    max_scale=2.0,
    step=0.1,
    mean_scale=1.0,
    std_scale=0.1,
    max_attempts=50,
    mode='gaussian'  # 'fixed', 'random', 'gaussian'
):
    base_h, base_w = base_mask.shape
    shape_h, shape_w = new_shape.shape
    base_bin = (base_mask > 20).astype(np.uint8)

    if mode == 'fixed':
        # 固定从大到小缩放
        scales = np.arange(max_scale, min_scale - step, -step)
    else:
        # 随机模式不需要提前生成 scales
        scales = [None] * max_attempts

    for _ in range(max_attempts):
        if mode == 'fixed':
            if not scales:
                break  # 没有更多scale可以尝试了
            scale = scales.pop(0)
        elif mode == 'random':
            scale = random.uniform(min_scale, max_scale)
        elif mode == 'gaussian':
            scale = np.random.normal(loc=mean_scale, scale=std_scale)
            scale = np.clip(scale, min_scale, max_scale)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        h, w = int(scale * shape_h), int(scale * shape_w)
        if h >= base_h or w >= base_w or h <= 5 or w <= 5:
            continue

        resized = Image.fromarray(new_shape).resize((w, h), Image.LANCZOS)
        shape = np.array(resized)
        shape_bin = (shape > 20).astype(np.uint8)

        x = random.randint(0, base_w - w)
        y = random.randint(0, base_h - h)
        region = base_bin[y:y + h, x:x + w]
        if not np.any(region & shape_bin):
            base_bin[y:y + h, x:x + w] |= shape_bin
            base_mask[y:y + h, x:x + w] = np.maximum(base_mask[y:y + h, x:x + w], shape)
            return True

    return False



def construct_combined_sketch(element_names, existing_mask, shape_library_root, image_size=512):
    sketch_mask = existing_mask.copy()

    if isinstance(element_names, str):
        element_names = [e.strip() for e in element_names.split(",") if e.strip()]

    for name in element_names:
        # category = name.lower().replace(" ", "_")
        category = name
        shape_folder = os.path.join(shape_library_root, category)
        if not os.path.exists(shape_folder):
            print(f"⚠️ 跳过未找到图库类别: {category}")
            continue

        try:
            shape_mask = load_random_canny(shape_folder)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            continue

        success = paste_nonoverlapping(sketch_mask, shape_mask, mode='random')
        if success:
            print(f"✅ 成功添加: {category}")
        else:
            print(f"❌ 无法放置: {category}，尝试失败")

    final_img = Image.fromarray(sketch_mask).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    return final_img


def load_mask_from_file(path, size):
    image = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    np_img = np.array(image)
    return standardize_canny(np_img)


def parse_args():
    parser = argparse.ArgumentParser(description="自动构建复合草图图像（统一风格）")
    parser.add_argument("--elements", type=str,default='dog, duck',help="用英文逗号分隔的元素类别（如 chess piece, book, vase）")
    parser.add_argument("--mask_file", type=str, default='E:\phd//3\code\ppi_control\data//admq\chickadee_canny.png' , help="已有元素草图 mask 图像路径")
    parser.add_argument("--shape_library", type=str, default="E:\phd//3\code\ppi_control\shape_library", help="元素图库根路径")
    parser.add_argument("--output_path", type=str, default="E:\phd//3\code\ppi_control\data//admq\chickadee_canny_fusion.png", help="输出图像保存路径")
    parser.add_argument("--image_size", type=int, default=512, help="输出图像尺寸")

    return parser.parse_args()

def construct_combined_sketch_from_args(args):
    base_mask = load_mask_from_file(args.mask_file, args.image_size)
    final_img = construct_combined_sketch(
        element_names=args.elements,
        existing_mask=base_mask,
        shape_library_root=args.shape_library,
        image_size=args.image_size
    )
    final_img.save(args.output_path)
    print(f"✅ 草图已保存: {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    construct_combined_sketch_from_args(args)
