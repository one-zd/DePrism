# 开始生成符合要求的新主函数脚本，支持：
# - 从 Excel 中逐条读取记录执行；
# - 支持 category 分类；
# - 每条记录出错后重试（间隔 1 分钟，最多 5 次）；
# - 跳过失败记录并记录日志。

import pandas as pd
import time
import logging
import argparse
import os
from run_all import main  # 假设原始函数名不变，作为 main(args) 使用
from argparse import Namespace
from tqdm import tqdm
def load_tasks_from_excel(excel_path, category_filter=None,case_number=0):
    df = pd.read_excel(excel_path)
    if category_filter:
        df = df[df['category'] == category_filter]
    df = df[case_number:]  # 从指定 case_number 开始执行
    return df.to_dict(orient='records')

def run_with_retry(row, max_retries=5, wait_seconds=60,dataset_path='', output_root='outputs', log_file='error_log.txt',scale=10,steps=50,seed=3,method="controlnet"):
    image_name = row['image_name']
    prompt = row['prompt']
    sensitive_words = row['sensitive_words']
    case_number = row['case_number']
    category = row['category']
    style_prompt = row['style_prompt']

    image_name_without_suffix = os.path.splitext(image_name)[0]
    image_path = os.path.join(dataset_path, image_name)
    output_path = os.path.join(output_root, image_name_without_suffix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    args_dict = {
        "image_path": image_path,
        "prompt": prompt,
        "sensitive_words": sensitive_words,
        "output_dir": output_path,
        "method": method,
        "a_prompt": style_prompt,
        "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        "scale": scale,
        "ddim_steps": steps,
        "strength": 1.0,
        "seed": seed,
        "low_threshold": 100,
        "high_threshold": 200,
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"\n✅ case: {case_number} |Image_name:{image_name} | Category: {category} | Try {retry_count + 1}")
            args = Namespace(**args_dict)
            main(args)
            return  # 成功后返回
        except Exception as e:
            retry_count += 1
            print(f"[ERROR] Case {case_number} failed on attempt {retry_count}. Error: {e}")
            if retry_count >= max_retries:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"Failed: case {case_number},Image_name:{image_name}, category {category}, prompt: {prompt}, error: {e}\n")
                print(f"❌ Skipped case {case_number} after {max_retries} retries.")
            else:
                time.sleep(wait_seconds)

def main_batch():
    parser = argparse.ArgumentParser(description="Batch runner for privacy-preserving generation")
    parser.add_argument("--excel_path", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100\multigen_100_key4.xlsx', help="Path to Excel file with task definitions")
    parser.add_argument("--dataset_path", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100//all')
    parser.add_argument("--output_root", type=str, default="E:\phd//3\code\ppi_control\outputs_dataset\outputs_controlnetplus_key4_cam0", help="Base output directory")
    parser.add_argument('--method', type=str, default='controlnet',
                        choices=['controlnet', 't2i_adapter','unicontrolnet','controlar'],
                        help='Choose the control t2i method')
    parser.add_argument("--category", type=str, default=None, help="Filter by category (optional)('other')")
    parser.add_argument("--case_number", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()
    tasks = load_tasks_from_excel(args.excel_path, category_filter=args.category,case_number=args.case_number)
    print(f"Loaded {len(tasks)} task(s) from Excel.")


    for row in tqdm(tasks, desc='Processing Images', unit='image'):
        run_with_retry(row,dataset_path=args.dataset_path, output_root=args.output_root, scale=args.scale, steps=args.steps, seed=args.seed, method=args.method)

if __name__ == "__main__":
    main_batch()
