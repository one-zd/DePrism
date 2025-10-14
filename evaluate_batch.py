import os
import pandas as pd
import torch
import numpy as np
import argparse
import lpips
from PIL import Image
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
from timm.models import create_model
import torch.nn.functional as F
from tqdm import tqdm
from NIMA.model.model import NIMA
from timm.models import vgg16
from collections import defaultdict
import glob
from torchmetrics.image import FID


# === 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='vgg').to(device)
clip_model = CLIPModel.from_pretrained("E:\phd//3\code\ppi_control\openai\clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("E:\phd//3\code\ppi_control\openai\clip-vit-large-patch14")

base_model = vgg16(pretrained=True)
nima_model = NIMA(base_model).to(device)
nima_model.load_state_dict(torch.load('E:\phd//3\code\ppi_control//NIMA\snapshots\epoch-82.pth'))  # 替换为实际路径
nima_model.eval()

global_fid_metric = FID(feature=2048).to(device)
category_fid_metrics = defaultdict(lambda: FID(feature=2048).to(device))
# === 核心指标函数
def compute_lpips(img1, img2):
    t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return lpips_fn(t(img1).unsqueeze(0).to(device), t(img2).unsqueeze(0).to(device)).item()

def compute_ssim(img1, img2):
    return ssim(np.array(img1), np.array(img2), channel_axis=-1, data_range=255)

def compute_psnr(img1, img2):
    return psnr(np.array(img1), np.array(img2), data_range=255)

def compute_clip_score(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    return outputs.logits_per_image.softmax(dim=1)[0][0].item()

def compute_nima_score(img):
    t = T.Compose([T.Resize(256),T.CenterCrop(224), T.ToTensor()])
    tensor = t(img).unsqueeze(0).to(device)
    pred = nima_model(tensor)
    scores = pred.squeeze().softmax(dim=0).detach().cpu().numpy()
    return sum(i * s for i, s in enumerate(scores, 1))

def subjective_score(nima, clip, lpips_score):
    lpips_adj = 1 - min(lpips_score, 1.0)
    return 0.4 * nima / 10 + 0.4 * clip + 0.2 * lpips_adj



def update_fid_metrics(gen_img, ref_img, category):
    transform_fid = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor()
    ])

    gen_img_uint8 = (transform_fid(gen_img) * 255).byte()
    ref_img_uint8 = (transform_fid(ref_img) * 255).byte()

    gen_tensor = gen_img_uint8.unsqueeze(0).to(device)
    ref_tensor = ref_img_uint8.unsqueeze(0).to(device)

    global_fid_metric.update(gen_tensor, real=False)
    global_fid_metric.update(ref_tensor, real=True)

    category_fid_metrics[category].update(gen_tensor, real=False)
    category_fid_metrics[category].update(ref_tensor, real=True)

def compute_all_fid_scores():
    results = {}
    results["ALL"] = global_fid_metric.compute().item()
    for cat, metric in category_fid_metrics.items():
        results[cat] = metric.compute().item()
    return results



# === 主流程
def evaluate_excel(excel_path, generated_dir, ref_dir,generated_name='final_fused_img', output_csv="eval_results.csv", category=None, case_number=0):
    df = pd.read_excel(excel_path)
    if category:
        df = df[df['category'] == category]
    df = df[case_number:]

    results = []
    category_stats = defaultdict(list)

    # 遍历所有图像样本
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['image_name']
        prompt = row['prompt']
        case = row['case_number']
        cat = row['category']
        ###不同文件夹读已有的固定命名的图像，生成的最终图像，gt图像####
        # gen_path = os.path.join(generated_dir, os.path.splitext(fname)[0], f"{generated_name}.png")
        #####同一个文件夹中不同图像名，原始图像####
        ref_path = os.path.join(ref_dir, fname)

        # gen_path = os.path.join(generated_dir, os.path.splitext(fname)[0], f"background_canny.png")         ##############background_canny , final_fused_img
        # ref_path = os.path.join(generated_dir, os.path.splitext(fname)[0], f"controlnet_background.png")      ###########controlnet_background, controlnet_gt

        ####找不同文件夹读部分固定命名的图像，敏感元素的canny图# canny_#和生成图#controlnet_camouflage_##controlnet_camouflage_###
        pattern = os.path.join(generated_dir,os.path.splitext(fname)[0], 'canny_*')
        # 使用glob模块查找所有匹配的文件
        image_files = glob.glob(pattern)
        if image_files:
            # 只读取第一个图像文件
            gen_path = image_files[0]
            # print("读取的图像文件：", os.path.basename(ref_path))
        else:
            print("未找到以 'controlnet_camouflage_' 开头的图像文件")
        # #####找不同文件夹读部分固定命名的图像，敏感元素的canny图和生成图###


        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            print(fname)
            raise FileNotFoundError("Missing file")

        gen_img = Image.open(gen_path).convert("RGB").resize((512, 512))
        ref_img = Image.open(ref_path).convert("RGB").resize((512, 512))

        p = compute_psnr(gen_img, ref_img)
        s = compute_ssim(gen_img, ref_img)
        l = compute_lpips(gen_img, ref_img)
        c = compute_clip_score(gen_img, prompt)
        n = compute_nima_score(gen_img)
        f = subjective_score(n, c, l)
        update_fid_metrics(gen_img, ref_img, cat)


        record = {
            "case_number": case,
            "category": cat,
            "image_name": fname,
            "PSNR": round(p, 2),
            "SSIM": round(s, 4),
            "LPIPS": round(l, 4),
            "CLIPScore": round(c, 4),
            "NIMA": round(n, 2),
            "SubjectiveScore": round(f, 4)
        }
        results.append(record)
        category_stats[cat].append(record)

    fid_scores = compute_all_fid_scores()
    print(f"📐 Global FID Score: {fid_scores['ALL']:.4f}")

    # 保存主CSV
    output_path_csv = os.path.join(generated_dir, f"{generated_name}{output_csv}")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path_csv, index=False)
    print(f"✅ Saved evaluation results to: {output_path_csv}")


    # 计算并输出各类均值和全局均值
    summary_rows = []
    for cat, items in category_stats.items():
        df_cat = pd.DataFrame(items)
        means = df_cat[["PSNR", "SSIM", "LPIPS", "CLIPScore", "NIMA", "SubjectiveScore"]].mean()
        means["FID"] = fid_scores.get(cat, None)
        means["category"] = cat
        summary_rows.append(means)

    # 添加全局均值
    df_all = pd.DataFrame(results)
    global_means = df_all[["PSNR", "SSIM", "LPIPS", "CLIPScore", "NIMA", "SubjectiveScore"]].mean()
    global_means["FID"] = fid_scores["ALL"]
    global_means["category"] = "ALL"
    summary_rows.append(global_means)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(generated_dir, f"{generated_name}_summary.csv")
    df_summary = df_summary[["category", "PSNR", "SSIM", "LPIPS", "CLIPScore", "NIMA", "SubjectiveScore", "FID"]]
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"📊 Saved category-wise summary to: {summary_csv_path}")



# === 命令行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image quality evaluator")
    parser.add_argument("--excel_path", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100\multigen_100_key4.xlsx')
    parser.add_argument("--generated_dir", type=str,default="E:\phd//3\code\ppi_control\outputs_dataset\outputs_controlnetplus_key4_cam0")
    parser.add_argument("--ref_dir", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100//all')
    parser.add_argument("--output_csv", type=str, default="eval_results.csv")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--case_number", type=int, default=0)
    parser.add_argument("--generated_name", type=str, default='ref2canny',help="final_fused_img,background_canny,controlnet_gt")
    args = parser.parse_args()

    evaluate_excel(
        excel_path=args.excel_path,
        generated_dir=args.generated_dir,
        ref_dir=args.ref_dir,
        output_csv=args.output_csv,
        category=args.category,
        generated_name=args.generated_name,
        case_number=args.case_number
    )
