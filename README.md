# 🧠 Not a Pandora’s Box Anymore: Preserving Privacy in Controllable Text-to-Image Inference  
隐私可控文本到图像生成的开源实现

---

## 📄 Overview | 项目简介

This repository provides the **official implementation** for the paper:  
**“Not a Pandora’s Box Anymore: Preserving Privacy in Controllable Text-to-Image Inference.”**

We propose a **privacy-preserving controllable generation pipeline**, integrating privacy protection into text-to-image diffusion inference.

本仓库是论文 **《Not a Pandora’s Box Anymore: Preserving Privacy in Controllable Text-to-Image Inference》** 的官方代码实现，提供了：
- 🧩 隐私保护可控图像生成（ControlNet, T2I Adapter 等方法）
- 📊 自动化图像质量评估（PSNR, SSIM, LPIPS, CLIP, NIMA, FID）

---

## 🧱 Repository Structure | 仓库结构

```
├── run_all_dataset.py       # 批量生成任务脚本（支持重试与日志）
├── evaluate_batch.py        # 自动化图像质量评估
├── run_all.py               # 图像生成核心函数（由 run_all_dataset 调用）
├── NIMA/                    # 图像美学质量模型（NIMA）
├── dataset/                 # 数据集示例（Excel + 原始图像）
└── outputs/                 # 生成结果保存目录
```

---

## ⚙️ Environment Setup | 环境配置

### 1️⃣ Create environment | 创建环境
```bash
conda create -n ppi_control python=3.10
conda activate ppi_control
```

### 2️⃣ Install dependencies | 安装依赖
```bash
pip install torch torchvision timm transformers lpips scikit-image pandas tqdm openpyxl torchmetrics pillow
```

> ⚠️ 若使用 GPU，请根据 CUDA 版本安装对应的 PyTorch。

---

## 📁 Dataset Preparation | 数据准备

Your dataset Excel file (e.g. `multigen_100_key4.xlsx`) should include:

| case_number | image_name | prompt | sensitive_words | category | style_prompt |
|--------------|-------------|--------|-----------------|-----------|---------------|
| 0 | img_001.png | A person in a park | face | human | cartoon style |

And your dataset directory:
```
dataset/
└── multigen-100/
    ├── all/
    │   ├── img_001.png
    │   ├── img_002.png
    │   └── ...
    └── multigen_100_key4.xlsx
```

---

## 🚀 Batch Generation | 批量隐私保护图像生成

Run the main generation script:

```bash
python run_all_dataset.py   --excel_path dataset/multigen-100/multigen_100_key4.xlsx   --dataset_path dataset/multigen-100/all   --output_root outputs/ppi_control_results   --method controlnet   --scale 10   --steps 50   --seed 3
```

### ✅ Features | 特性
- 支持多种可控生成方法：`controlnet`, `t2i_adapter`, `unicontrolnet`, `controlar`
- 自动重试与错误记录（最多 5 次）
- 按分类筛选执行：`--category portrait`
- 输出日志与错误记录文件

生成结果结构：
```
outputs/ppi_control_results/
└── img_001/
    ├── final_fused_img.png
    ├── canny_mask.png
    └── ...
```

---

## 📊 Evaluation | 自动化评估

Run the evaluation script after generation:

```bash
python evaluate_batch.py   --excel_path dataset/multigen-100/multigen_100_key4.xlsx   --generated_dir outputs/ppi_control_results   --ref_dir dataset/multigen-100/all   --generated_name final_fused_img   --output_csv eval_results.csv
```

### 📈 Metrics | 评估指标

| Metric | Description |
|:--------|:-------------|
| **PSNR** | Peak Signal-to-Noise Ratio |
| **SSIM** | Structural Similarity Index |
| **LPIPS** | Perceptual similarity (VGG-based) |
| **CLIP Score** | Text-image alignment |
| **NIMA** | Aesthetic score |
| **FID** | Fréchet Inception Distance |
| **Subjective Score** | Weighted fusion of NIMA + CLIP + LPIPS |

Output files:
- `eval_results.csv` → 每张图片的指标结果  
- `*_summary.csv` → 各类别与全局平均指标  

---

## 📊 Example Output | 示例输出

```
📐 Global FID Score: 12.3478
✅ Saved evaluation results to: outputs/final_fused_img_eval_results.csv
📊 Saved category-wise summary to: outputs/final_fused_img_summary.csv
```

---

## 🧮 Model Dependencies | 模型依赖

- **CLIP:** `openai/clip-vit-large-patch14`
- **NIMA:** VGG16 backbone, load from `NIMA/snapshots/epoch-82.pth`
- **Control methods:** integrated in `run_all.py`

请根据本地路径修改 `evaluate_batch.py` 中模型加载位置。

---

## 🧾 Citation | 引用

If you use this code, please cite:

```bibtex
@article{YourName2025PrivacyControlT2I,
  title={Not a Pandora’s Box Anymore: Preserving Privacy in Controllable Text-to-Image Inference},
  author={Zhangdong Wang, Tongqing Zhou, Zhihuang Liu, Jiaohua Qin and Zhiping Cai},
  journal={Submit to TIP},
  year={2025}
}
```

---

## 📜 License | 许可协议

This project is released **for academic research purposes only**.  
Commercial use requires prior permission from the authors.

本项目仅供学术研究使用，如需商业授权请联系论文作者。
