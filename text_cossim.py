import json
import os
import sys
import pandas as pd

from tqdm import tqdm
import numpy as np
import gc
from collections import defaultdict
import argparse

from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


from bert_score import score as bert_score
import Levenshtein

class TextSimEvaluator:
    def __init__(self, sb_model='E:\phd//3\code\ppi_control//all-MiniLM-L6-v2'):
        self.sb_model = SentenceTransformer(sb_model)

    def cosine_sim(self, texts1, texts2):
        emb1 = self.sb_model.encode(texts1, convert_to_tensor=True)
        emb2 = self.sb_model.encode(texts2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).diagonal().cpu().numpy()

    # def bertscore(self, preds, refs):
    #     P, R, F1 = bert_score(preds, refs,lang="en", rescale_with_baseline=True)
    #     return F1.numpy()

    def edit_distance(self, s1, s2):
        return Levenshtein.distance(s1, s2)

def sentence_sim(llm_dir, sim_model):
    original, reduced, camouflage = llm_text(llm_dir)

    o2b_cos = sim_model.cosine_sim([original], [reduced])[0]
    o2s_cos = sim_model.cosine_sim([original], [camouflage])[0]

    # o2b_bert = sim_model.bertscore([reduced], [original])[0]
    # o2s_bert = sim_model.bertscore([camouflage], [original])[0]

    o2b_edit = sim_model.edit_distance(original, reduced)
    o2s_edit = sim_model.edit_distance(original, camouflage)


    return {
        "o2b_cos": o2b_cos,
        "o2s_cos": o2s_cos,
        # "o2b_bert": o2b_bert,
        # "o2s_bert": o2s_bert,
        "o2b_edit": o2b_edit,
        "o2s_edit": o2s_edit,
    }


def llm_text(json_dir):
    """
    从指定目录中读取JSON，返回原句、简化句、伪装prompt。
    """
    base_name = "llm_text_sec_deepseek_32b_1.5"
    json_path = os.path.join(json_dir, f"{base_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON not found: {json_path}")

    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"❌ JSON empty: {json_path}")

    data = data[0] if isinstance(data, list) else data
    original_sentence = data.get("original_sentence", "")
    reduced_sentence = data.get("perturbed_sentence", {}).get("reduced_sentence", "")

    camouflage_prompt = ""
    camouflage_dict = data.get("perturbed_sentence", {}).get("camouflage", {})
    if isinstance(camouflage_dict, dict):
        for _, val in camouflage_dict.items():
            camouflage_prompt = val.get("camouflage prompt", "")
            break  # 默认只取第一个词的伪装prompt

    return original_sentence, reduced_sentence, camouflage_prompt


def evaluate_excel_text(excel_path, generated_dir, generated_name='textsim',output_csv="eval_results.csv", category=None, case_number=0):

    df = pd.read_excel(excel_path)
    if category:
        df = df[df['category'] == category]
    df = df[case_number:]

    results = []
    category_stats = defaultdict(list)
    sim_model = TextSimEvaluator()

    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
        fname = row['image_name']
        prompt = row.get('prompt', '')
        cat = row.get('category', 'unknown')
        case = row.get('case_number', 0)

        json_dir = os.path.join(generated_dir, os.path.splitext(fname)[0])
        sim_metrics = sentence_sim(json_dir, sim_model)

        if sim_metrics is None:
            continue

        record = {
            "image_name": fname,
            "prompt": prompt,
            "category": cat,
            "case_number": case,
            **sim_metrics
        }

        results.append(record)
        category_stats[cat].append(record)

    df_results = pd.DataFrame(results)
    output_path_csv = os.path.join(generated_dir, f"{generated_name}_{output_csv}")
    df_results.to_csv(output_path_csv, index=False)
    print(f"\n✅ Saved {len(results)} results to: {output_path_csv}")
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} samples due to read errors or missing fields.")

    # 分类和全局均值
    summary_rows = []
    for cat, items in category_stats.items():
        df_cat = pd.DataFrame(items)
        mean_metrics = df_cat[
            # ["o2b_cos", "o2s_cos", "o2b_bert", "o2s_bert", "o2b_edit", "o2s_edit"]
            ["o2b_cos", "o2s_cos", "o2b_edit", "o2s_edit"]
        ].mean()
        mean_metrics["category"] = cat
        summary_rows.append(mean_metrics)

    df_all = pd.DataFrame(results)
    global_means = df_all[
        # ["o2b_cos", "o2s_cos", "o2b_bert", "o2s_bert", "o2b_edit", "o2s_edit"]
        ["o2b_cos", "o2s_cos", "o2b_edit", "o2s_edit"]
    ].mean()
    global_means["category"] = "ALL"
    summary_rows.append(global_means)

    summary_csv_path = os.path.join(generated_dir, f"{generated_name}_sim_summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    # df_summary = df_summary[["category", "o2b_cos", "o2s_cos", "o2b_bert", "o2s_bert", "o2b_edit", "o2s_edit"]]
    df_summary = df_summary[["category", "o2b_cos", "o2s_cos", "o2b_edit", "o2s_edit"]]
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"📊 Saved category-wise similarity summary to: {summary_csv_path}")

    # === 分布图可视化 ===
    # 相似度分布图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_all["o2b_cos"], label="Original → Reduced (S-BERT)", color='blue')
    sns.kdeplot(df_all["o2s_cos"], label="Original → Camouflage (S-BERT)", color='orange')
    plt.title("Sentence-BERT Cosine Similarity")
    plt.legend()
    plt.savefig(os.path.join(generated_dir, f"{generated_name}_sb_similarity_dist.png"))
    plt.close()

    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(df_all["o2b_bert"], label="Original → Reduced (BERTScore)", color='green')
    # sns.kdeplot(df_all["o2s_bert"], label="Original → Camouflage (BERTScore)", color='red')
    # plt.title("BERTScore F1 Distribution")
    # plt.legend()
    # plt.savefig(os.path.join(generated_dir, f"{generated_name}_bertscore_dist.png"))
    # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default='E:\phd//3\code\ppi_control\dataset\multigen-100\multigen_100.xlsx')
    parser.add_argument("--generated_dir", type=str,default="E:\phd//3\code\ppi_control\outputs_controlnet")
    parser.add_argument("--generated_name", type=str, default='textsim')
    parser.add_argument("--output_csv", type=str, default="eval_results.csv")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--case_number", type=int, default=0)
    args = parser.parse_args()

    evaluate_excel_text(
        excel_path=args.excel_path,
        generated_dir=args.generated_dir,
        generated_name=args.generated_name,
        output_csv=args.output_csv,
        category=args.category,
        case_number=args.case_number
    )
