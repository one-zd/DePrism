import os
import torch
import lpips
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
from timm.models import create_model
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. LPIPS 模型
lpips_fn = lpips.LPIPS(net='vgg').to(device)

# === 2. CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === 3. NIMA 模型 (建议 MobileNet backbone)
nima_model = create_model('mobilenetv2_100', pretrained=False, num_classes=10)
nima_model.load_state_dict(torch.load('nima_mobilenet.pth'))  # 需准备该模型
nima_model.eval().to(device)

def compute_lpips(img1, img2):
    t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    d = lpips_fn(t(img1).unsqueeze(0).to(device), t(img2).unsqueeze(0).to(device))
    return d.item()

def compute_ssim(img1, img2):
    return ssim(np.array(img1), np.array(img2), multichannel=True, data_range=255)

def compute_psnr(img1, img2):
    return psnr(np.array(img1), np.array(img2), data_range=255)

def compute_clip_score(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.softmax(dim=1)[0][0].item()

def compute_nima_score(img):
    tensor = to_tensor(img).unsqueeze(0).to(device)
    pred = nima_model(tensor)
    scores = pred.squeeze().softmax(dim=0).detach().cpu().numpy()
    return sum(i * s for i, s in enumerate(scores, 1))

def subjective_score(nima, clip, lpips_score):
    lpips_adj = 1 - min(lpips_score, 1.0)
    return 0.4 * nima / 10 + 0.4 * clip + 0.2 * lpips_adj

# === 主流程
def evaluate_all(gen_dir, ref_dir, prompt_file, output_csv="results.csv"):
    df_prompt = pd.read_csv(prompt_file)
    results = []

    for idx, row in df_prompt.iterrows():
        fname, prompt = row['filename'], row['prompt']
        gen_path = os.path.join(gen_dir, fname)
        ref_path = os.path.join(ref_dir, fname)

        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            continue

        gen_img = Image.open(gen_path).convert("RGB").resize((512, 512))
        ref_img = Image.open(ref_path).convert("RGB").resize((512, 512))

        p = compute_psnr(gen_img, ref_img)
        s = compute_ssim(gen_img, ref_img)
        l = compute_lpips(gen_img, ref_img)
        c = compute_clip_score(gen_img, prompt)
        n = compute_nima_score(gen_img)
        fused = subjective_score(n, c, l)

        results.append({
            "filename": fname,
            "PSNR": round(p, 2),
            "SSIM": round(s, 4),
            "LPIPS": round(l, 4),
            "CLIPScore": round(c, 4),
            "NIMA": round(n, 2),
            "SubjectiveScore": round(fused, 4)
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Done. Saved to {output_csv}")

# 用法示例
# evaluate_all("inputs", "refs", "prompts.csv")


