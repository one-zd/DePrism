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
---

## 🧱 Repository Structure | 仓库结构

```
├── scripts/
│   ├── run_all_dataset.py              # 批量生成任务（支持重试与日志）
│   ├── run_all.py                      # 单样本 DePrism 核心流程
│   ├── evaluate_comprehensive.py       # 统一评测器
│   ├── controlnet.py                   # 可控生成方法接口
│   ├── canny_fusion.py                 # 伪装边缘组合
│   └── final_image_fusion.py           # 客户端图像融合
├── NIMA/                               # NIMA 模型定义与 checkpoint
├── dataset/                            # Excel/CSV 元数据和原始图像
├── outputs_dataset/                    # 各方法生成结果
└── openai/clip-vit-large-patch14/      # 本地 CLIP 模型
```

---

## ⚙️ Environment Setup | 环境配置

### 1️⃣ Create environment | 创建环境

```bash
conda env create -f scripts/environment.yml
conda activate control
```

`environment.yml` 保存的论文实验环境基于 Python 3.8、PyTorch 1.12 和 CUDA 11.3。若只安装评测依赖，可使用：

### 2️⃣ Install evaluation dependencies | 安装评测依赖

```bash
pip install torch torchvision transformers lpips scikit-image pandas tqdm openpyxl torchmetrics torch-fidelity pillow tabulate
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
python scripts/run_all_dataset.py \
  --excel_path dataset/multigen-100/multigen_100_key4.xlsx \
  --dataset_path dataset/multigen-100/all \
  --output_root outputs_dataset/outputs_controlnetplus_key4 \
  --method controlnet \
  --scale 10 \
  --steps 50 \
  --seed 3
```

### ✅ Features | 特性
- 支持多种可控生成方法：`controlnet`, `t2i_adapter`, `unicontrolnet`, `controlar`
- 自动重试与错误记录（最多 5 次）
- 按分类筛选执行：`--category portrait`
- 输出日志与错误记录文件

生成结果结构：
```
outputs_dataset/outputs_controlnetplus_key4/
└── img_001/
    ├── background_canny.png
    ├── controlnet_background.png
    ├── controlnet_camouflage_<keyword>.png
    ├── controlnet_gt.png
    └── final_fused_img.png
```

---

## 📊 Comprehensive Evaluation | 全面统一评测

使用 `scripts/evaluate_comprehensive.py` 对 Excel/CSV 元数据执行批量评测。评测器支持分类汇总、多方法比较、L2-normalized CLIP embedding 余弦相似度和 NIMA-5C，并保存完整的逐样本结果与实验日志。

### 单方法评测

下面的命令同时评测最终 DePrism 输出相对于原始图像和 Standard 输出的质量：

```bash
python scripts/evaluate_comprehensive.py \
  --metadata dataset/multigen-100/multigen_100.xlsx \
  --ref-dir dataset/multigen-100/all \
  --generated-dir outputs_dataset/outputs_controlnet \
  --method-name ControlNet \
  --generated-name final_fused_img.png \
  --comparison-name controlnet_gt.png \
  --comparison-label standard \
  --output-dir scripts/comprehensive_evaluation \
  --output-prefix controlnet_final
```

### 多方法统一评测

使用多个 `--run LABEL=PATH` 可在一次运行中评测和汇总不同方法：

```bash
python scripts/evaluate_comprehensive.py \
  --metadata dataset/multigen-100/multigen_100.xlsx \
  --ref-dir dataset/multigen-100/all \
  --run ControlNet=outputs_dataset/outputs_controlnet \
  --run T2I-Adapter=outputs_dataset/outputs_t2i_adapter \
  --run Uni-ControlNet=outputs_dataset/outputs_unicontrolnet \
  --run ControlNet++=outputs_dataset/outputs_controlnet_plusplus \
  --generated-name final_fused_img.png \
  --comparison-name controlnet_gt.png \
  --output-dir scripts/comprehensive_evaluation \
  --output-prefix table2_methods
```

### 图像对与指标命名

评测器明确区分两套图像对，避免将隐私和效用口径混淆：

| 输出列前缀 | 图像对 | 常见用途 |
| --- | --- | --- |
| `ref_*`、`clip_i_ref` | 生成图 vs 原始参考图 | 最终图像效用或输入侧隐私 |
| `standard_*`、`clip_i_standard` | 生成图 vs `--comparison-name` | DePrism vs Standard 输出一致性 |
| `clip_t_prompt` | 生成图 vs 原始 prompt | 文本—图像语义对齐 |
| `clip_t_prompt_style` | 生成图 vs style prompt + prompt | 风格条件下的语义对齐 |

`--comparison-name` 和 `--comparison-label` 可用于任意额外图像对。例如评测最终图像与上下文分支时，可使用：

```bash
--comparison-name controlnet_background.png --comparison-label context
```

### 评估指标

| Metric | Description | Direction |
| --- | --- | --- |
| **PSNR** | 像素级峰值信噪比 | 相似性任务通常越高越好 |
| **SSIM** | 结构相似性 | 相似性任务通常越高越好 |
| **LPIPS** | VGG 感知距离 | 相似性任务通常越低越好 |
| **FID** | 数据集级生成/参考分布距离 | 越低越好；不应解释单样本 FID |
| **CLIP-T** | L2-normalized text/image embeddings 的余弦相似度 | 语义对齐通常越高越好 |
| **CLIP-I** | 两幅图像的归一化 CLIP embeddings 余弦相似度 | 方向取决于具体隐私/效用图像对 |
| **NIMA-5C** | ImageNet 归一化、五裁剪后的美学分布与均值 | 越高通常表示美学质量越好 |

评测器分别报告每项标准指标，不使用人为加权的综合分数。

### NIMA 评测模式

默认使用 NIMA-5C：

```bash
--nima-variants five_crop_norm
```

若要执行预处理审计，可同时计算四个版本：

```bash
--nima-variants \
  no_norm_center_crop \
  center_crop_norm \
  resize_direct_norm \
  five_crop_norm
```

可通过 `--nima-targets` 控制 NIMA 的计算对象：

```bash
--nima-targets generated reference comparison
```

### 输出文件

以 `--output-prefix comprehensive` 为例，评测器生成：

| 文件 | 内容 |
| --- | --- |
| `comprehensive_per_sample.csv` | 每个样本、每个方法的完整指标 |
| `comprehensive_summary.csv` | 按方法和类别统计的均值、标准差及 FID |
| `comprehensive_missing.csv` | 缺失文件或多重 glob 匹配记录 |
| `comprehensive_report.md` | 可读的实验配置、指标说明与汇总表 |
| `comprehensive_log.json` | 路径、模型、版本、样本数量和输出文件日志 |

缺失图像默认记录后跳过；需要严格复现时可使用：

```bash
--missing-policy error
```

模型默认只从本地读取。仅在明确需要联网下载 CLIP 时使用：

```bash
--allow-model-download
```

### 快速检查与指标开关

可以关闭较慢的模型指标，用于检查元数据和图像路径：

```bash
python scripts/evaluate_comprehensive.py \
  --metadata dataset/multigen-100/multigen_test.xlsx \
  --ref-dir dataset/multigen-100/all \
  --generated-dir outputs_dataset/outputs_controlnet \
  --generated-name final_fused_img.png \
  --limit 2 \
  --skip-clip --skip-nima --skip-lpips --skip-fid
```

## 🧮 Model Dependencies | 模型依赖

- **CLIP:** `openai/clip-vit-large-patch14`
- **NIMA:** VGG16 backbone，默认 checkpoint 为 `NIMA/snapshots/epoch-82.pth`
- **Control methods:** integrated in `run_all.py`

评测器支持通过 `--clip-model` 和 `--nima-ckpt` 指定路径；相对路径以仓库根目录解析，不需要修改源代码。

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
