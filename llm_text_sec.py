import os
import sys
import json
import time
import random
import logging
import traceback
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import re



def set_random_seed(seed_value=4):
    """
    固定随机种子以确保结果可复现。
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def parse_args():
    """
    Parse command line arguments for model, temperature settings, and prompt input
    """
    parser = argparse.ArgumentParser(description="Local LLM Paraphrasing for Privacy Protection")

    # Model argument
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-r1:32b",
        help="Model name to use for paraphrasing (default: deepseek-r1:32b)"
    )

    # Temperature argument - can be a single value or multiple values
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[1.5],
        help="Temperature value(s) for text generation (default: 0.5,1.0, 1.5)"
    )

    # Maximum number of examples to process (optional)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: process all)"
    )

    # API endpoint
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:3000/ollama/api/chat",
        help="API endpoint for the local LLM (default: http://localhost:3000/ollama/api/chat)"
    )

    # Prompt input
    parser.add_argument(
        "--prompt",
        type=str,
        default="The living room has a large sofa, a wooden coffee table, and a flat - screen TV mounted on the wall. ",
        help="Input prompt for paraphrasing"
    )

    # Sensitive words input
    parser.add_argument(
        "--sensitive_words",
        type=str,
        default="sofa,table",
        help="Comma-separated list of sensitive words"
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="E:/phd/3/code/ppi_control/data/text",
        help="Directory to save output JSON/CSV results"
    )


    return parser.parse_args()


class LocalLLMParaphraser:
    def __init__(self, base_url="", api_key=None):
        """
        Initialize with the Ollama API endpoint.
        """
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Test connection
        self.test_connection()

    def test_connection(self):
        """
        Test the connection to the Ollama server
        """
        try:
            # Simple request to check if server is running
            api_version_url = self.base_url.replace("/api/chat", "/api/version")
            api_version_url = api_version_url.replace("ollama/api/chat", "api/version")  # Handle different URL patterns
            response = requests.get(api_version_url)
            if response.status_code == 200:
                print(f"Successfully connected to Ollama server: {response.json()}")
            else:
                print(f"Warning: Could not connect to Ollama server. Status: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error connecting to Ollama server: {e}")
            print("Please make sure the Ollama server is running and accessible.")

    def define_system_message(self):
        """
        Define the system prompt for privacy-preserving text paraphrasing.
        """
        return (
            "你是一个智能文本处理系统，负责对输入句子进行敏感词保护处理，并严格按照示例的JSON格式输出。用户输入包括原始句子和多个敏感词（需要保护的物体名称）。"
            "你需要根据以下规则生成结构化输出："
            "1. 每个敏感词生成一条记录，包含以下字段："
            "   - \"camouflage prompt\": 仅保留敏感词，替换其他物体词后的完整句子。注意是保留敏感词，替换其他主体词！"
            "   - \"camouflage element\": 替换后句子中除敏感词外的其他主体物体词（数组）。主体词是指具体可见物体（如 sofa、table、book），不包括背景词（如 wall、sky）"
            "2. 在所有敏感词处理完毕后，还需统一生成一个字段："
            "   - \"reduced_sentence\": 从原始句子中移除所有敏感词后的句子，保持语法正确。如果句子中已无其他主体词，则随机添加一个相关场景的主体词。"
            "3. 输出结构必须为一个 JSON 对象，包含二个顶级字段："
            "   - \"camouflage\": 一个对象，键为每个敏感词，值为对应处理后的句子和物体"
            "   - \"reduced_sentence\": 统一的敏感词移除结果"
            "4. 所有字段必须严格遵循格式，不允许额外的注释或解释性文字，输出为标准 JSON 格式。"

            '示例输入：{"original_sentence": "The living room has a large sofa, a wooden coffee table, and a flat - screen TV mounted on the wall.", '
            '"sensitive_words": ["sofa", "table"]}'

            '示例输出（必须严格遵循此格式）：'
            '{'
            '  "camouflage": {'
            '    "sofa": {'
            '      "camouflage prompt": "The living room has a large sofa, a glass dining table, and a home theater system mounted on the wall.",'
            '      "camouflage element": ["glass dining table", "home theater system"]'
            '    },'
            '    "table": {'
            '      "camouflage prompt": "The living room has a leather armchair, a glass dining table, and an LCD monitor mounted on the wall.",'
            '      "camouflage element": ["leather armchair", "LCD monitor"]'
            '    }'
            '  },'
            '  "reduced_sentence": "a flat - screen TV mounted on the wall.",'
            '}'
            
            "⚠️ 重要说明：你必须严格按照示例格式输出 JSON 格式数据，不允许任何额外的文本、解释或格式变化！"
            "⚠️ 重要说明：你必须严格按照示例格式输出 JSON 格式数据，不允许任何额外的文本、解释或格式变化！"
            "⚠️ 重要说明：你必须严格按照示例格式输出 JSON 格式数据，不允许任何额外的文本、解释或格式变化！"
        )



    def paraphrase(self, text, model="deepseek-r1:32b", temperature=0.75):
        """
        Paraphrase the given text using the local LLM.
        """
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.define_system_message()},
                {"role": "user", "content": f"{text}"}
            ],
            "stream": False,
            "temperature": temperature
        }

        try:
            start_time = time.time()
            response = requests.post(self.base_url, json=data, headers=self.headers)
            processing_time = time.time() - start_time

            if response.status_code == 200:
                # Extract content from the response
                try:
                    response_json = response.json()

                    # # 如果 response_json 已经是 dict，不需要再用 json.loads()
                    # if isinstance(response_json, dict):
                    #     return response_json, processing_time
                    # elif isinstance(response_json, str):
                    #     # 处理字符串格式的 JSON
                    #     response_json = json.loads(response_json)
                    #     return response_json, processing_time

                    if "message" in response_json and "content" in response_json["message"]:

                        content = response_json["message"]["content"].strip()

                        # Handle DeepSeek's thinking process
                        if "<think>" in content and "</think>" in content:
                            # Extract only the text after the thinking process
                            parts = content.split("</think>", 1)
                            if len(parts) > 1:
                                # Get the text after </think>
                                content = parts[1].strip()
                            else:
                                # If format is unusual, just remove the thinking tags
                                content = content.replace("<think>", "").replace("</think>", "").strip()

                        if "```json" in content and "```" in content:
                            content = re.sub(r"```json\s*", "", content)  # 去掉开头 ```json
                            content = content.rstrip("```")  # 去掉结尾 ```

                        # print(content,'11111111111')    #######输出{
                        #   "chess piece": "a chess piece and a book on the shelf, a vase on the stand.",
                        #   "clock": "a desk lamp and a radio on the counter, a clock on the stand."
                        # }
                        return content, processing_time
                    else:
                        print(f"Unexpected response structure: {response_json}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON response: {response.text}")

                return None, processing_time
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None, processing_time
        except Exception as e:
            print(f"Request error: {str(e)}")
            traceback.print_exc()
            return None, 0


def process_single_prompt(prompt, sensitive_words, paraphraser, model, temperature):
    """
    Process a single prompt with sensitive words using the paraphraser.
    """
    try:
        # Prepare the input data
        input_data = {
            "model": model,
            "messages": [
                {"role": "system", "content": paraphraser.define_system_message()},
                {"role": "user", "content": json.dumps({
                    "original_sentence": prompt,
                    "sensitive_words": sensitive_words
                })}
            ],
            "stream": False,
            "temperature": temperature
        }

        paraphrased_text, processing_time = paraphraser.paraphrase(input_data)

        if paraphrased_text:
            # Parse the JSON response
            try:
                # paraphrased_json = json.loads(paraphrased_text)
                if isinstance(paraphrased_text, dict):
                    paraphrased_json = paraphrased_text  # 直接用 dict
                else:
                    paraphrased_json = json.loads(paraphrased_text)  # 解析 JSON 字符串

                # paraphrased_json = json.dumps(paraphrased_text, ensure_ascii=False)

                result = {
                    "original_sentence": prompt,
                    "sensitive_words": sensitive_words,
                    "perturbed_sentence": paraphrased_json,
                    "processing_time": processing_time
                }
                return result, processing_time
            except json.JSONDecodeError:
                print(f"Invalid JSON response: {paraphrased_text}")
                return None, 0
        else:
            print(f"Failed to paraphrase prompt: {prompt}")
            return None, 0
    except Exception as e:
        print(f"Error processing prompt: {str(e)}")
        traceback.print_exc()
        return None, 0


def save_results(results, temperature, model, avg_time, output_dir):
    """
    Save the paraphrasing results to JSON and CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if model == "llama3:latest":
        model_name = "llama3_8b"
    elif model == "deepseek-r1:14b":
        model_name = "deepseek_14b"
    elif model == "deepseek-r1:32b":
        model_name = "deepseek_32b"
    else:
        model_name = model.replace(":", "_")

    base_name = f"llm_text_sec_{model_name}_{temperature}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    csv_path = os.path.join(output_dir, f"{base_name}.csv")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"Results saved to {json_path} and {csv_path}")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Number of examples processed: {len(results)}")





def main():
    # Parse command line arguments
    args = parse_args()

    set_random_seed(42)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Check if prompt is provided
    if not args.prompt:
        logger.error("No prompt provided. Please use --prompt to specify the input text.")
        return

    # Log selected parameters
    logger.info(f"Using model: {args.model}")
    logger.info(f"Temperature value(s): {args.temperature}")

    # Initialize the paraphraser
    paraphraser = LocalLLMParaphraser(base_url=args.api_url, api_key="sk-890d6329f8d4470c8457050674484e70")

    try:
        for temperature in args.temperature:
            logger.info(f"Processing prompt with temperature = {temperature}")

            try:
                result, processing_time = process_single_prompt(
                    args.prompt,
                    args.sensitive_words,
                    paraphraser,
                    args.model,
                    temperature,
                )

                if result:
                    # Save the results
                    save_results([result], temperature, args.model, processing_time, args.output_dir)

                    logger.info(f"Completed processing with temperature = {temperature}")
                else:
                    logger.error(f"Failed to paraphrase prompt with temperature = {temperature}")
            except KeyboardInterrupt:
                logger.info(f"Processing for temperature {temperature} interrupted by user.")
                raise  # Re-raise to be caught by the outer try-except
            except Exception as e:
                logger.error(f"Error processing with temperature {temperature}: {e}")
                logger.error(traceback.format_exc())
                continue

        logger.info("All processing completed")


        # Release resources
        # release_resources()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user. Results up to this point have been saved.")
        # release_resources()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        # release_resources()
        sys.exit(1)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()