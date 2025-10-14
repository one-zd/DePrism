import cv2
import numpy as np
import os
import argparse


def feather_blend(foreground, background, mask, blur_ksize=31, dilate_kernel=5, blur_sigma=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    feather_mask = cv2.GaussianBlur(mask_dilated, (blur_ksize, blur_ksize), sigmaX=blur_sigma)

    alpha = feather_mask.astype(np.float32) / 255.0
    alpha = cv2.merge([alpha]*3)

    blended = foreground.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)


def poisson_blend(foreground, background, mask, mode="normal"):
    ys, xs = np.where(mask > 0)
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    center = (center_x, center_y)

    flags = cv2.NORMAL_CLONE if mode == "normal" else cv2.MIXED_CLONE
    return cv2.seamlessClone(foreground, background, mask, center, flags)


def run_privacy_fusion(
    base_image,                 # 背景图路径(str) 或 图像数据(ndarray)
    camouflage_images,          # 前景图路径(str) 或 图像数据(ndarray)
    mask_image,                 # mask路径(str) 或 图像数据(ndarray)
    output_dir=None,             # 保存文件夹（可选）
    intermediate=False,          # 是否保存中间融合结果（bool）
    method="feather",            # "feather" 或 "poisson"
    poisson_mode="normal",       # 泊松融合子模式
    dilate_kernel=1,
    blur_ksize=11,
    blur_sigma=10,
    step_idx=[0]                 # 当前融合步数（用列表传引用以自增）
):
    """
    运行隐私融合，支持路径或数组输入，支持中间步骤保存。
    """

    def load_image(x, mode="color"):
        if isinstance(x, str):
            if mode == "color":
                img = cv2.imread(x, cv2.IMREAD_COLOR)
            elif mode == "grayscale":
                img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if img is None:
                raise FileNotFoundError(f"无法读取图像：{x}")
            return img
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise TypeError(f"输入类型错误，期望 str 或 ndarray，但收到 {type(x)}")

    # 加载图像
    fg = load_image(camouflage_images, mode="color")
    bg = load_image(base_image, mode="color")
    mask = load_image(mask_image, mode="grayscale")

    # 校验尺寸一致性
    if fg.shape[:2] != mask.shape:
        raise ValueError(f"前景图尺寸 {fg.shape[:2]} 和 mask尺寸 {mask.shape} 不一致！")
    if fg.shape[:2] != bg.shape[:2]:
        raise ValueError(f"前景图尺寸 {fg.shape[:2]} 和 背景图尺寸 {bg.shape[:2]} 不一致！")

    # 选择融合方法
    if method == "poisson":
        result = poisson_blend(fg, bg, mask, mode=poisson_mode)
    else:  # feather
        result = feather_blend(fg, bg, mask,
                               blur_ksize=blur_ksize,
                               dilate_kernel=dilate_kernel,
                               blur_sigma=blur_sigma)

    # 如果需要保存中间步骤
    if intermediate and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'intermediate_fused_step{step_idx[0]}.png')
        cv2.imwrite(output_path, result)
        print(f"✅ Step {step_idx[0]} 融合完成，保存到 {output_path}")
        step_idx[0] += 1  # 更新步数

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="融合前景图到背景图（支持 feather / poisson）")

    parser.add_argument("--foreground", type=str, help="前景图路径（伪装图）")
    parser.add_argument("--background", type=str, help="背景图路径")
    parser.add_argument("--mask", type=str,  help="mask 图路径（255表示保留）")
    parser.add_argument("--output_dir", type=str,  help="输出路径")

    parser.add_argument("--method", type=str, default="feather", choices=["poisson", "feather"], help="融合方法")
    parser.add_argument("--poisson_mode", type=str, default="normal", choices=["normal", "mixed"], help="泊松融合方式")
    parser.add_argument("--dilate_kernel", type=int, default=5, help="膨胀核大小")
    parser.add_argument("--blur_ksize", type=int, default=31, help="高斯模糊核大小（必须为奇数）")
    parser.add_argument("--blur_sigma", type=float, default=10, help="高斯模糊 sigma")

    return parser.parse_args()


def main():
    args = parse_args()

    run_privacy_fusion(
        base_image=args.background,
        camouflage_images=args.foreground,
        mask_image=args.mask,
        output_dir=args.output_dir,
        method=args.method,
        poisson_mode=args.poisson_mode,
        dilate_kernel=args.dilate_kernel,
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma
    )


if __name__ == "__main__":
    main()
