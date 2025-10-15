

import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import multiprocessing
import numpy as np
import time
import argparse
import tempfile

def build_prompt(tokenizer, question: str, options: dict):
    """构建标准的评估Prompt。"""
    options_str = "\n".join([f"{key}. {value}" for key, value in sorted(options.items())])
    prompt_text = (
        f"你是一位顶级的医学专家。请仔细分析以下医学多项选择题，并直接选择最正确的选项字母。\n\n"
        f"问题: {question}\n\n"
        f"选项:\n{options_str}\n\n"
        f"正确答案的选项是："
    )
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_option_letter(reply: str, options: dict):
    """从模型回复中稳健地提取选项字母。"""
    r = reply.strip().upper()
    for k in options:
        if r.startswith(k): return k
    for k in options:
        if k in r: return k
    return ""

def load_model_for_worker(model_dir: Path, device: str):
    """为单个工作进程加载模型到指定设备。"""
    print(f"[INFO] Process-{os.getpid()} is loading model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def inference_worker(gpu_id: int, data_chunk: list, model_path: Path, max_new_tokens: int, worker_queue):
    """在单个GPU上运行推理的工作函数，并将结果放入队列。"""
    device = f"cuda:{gpu_id}"
    try:
        tokenizer, model = load_model_for_worker(model_path, device)
        
        for item in tqdm(data_chunk, desc=f"GPU-{gpu_id}", position=gpu_id, unit="条"):
            prompt = build_prompt(tokenizer, item["question"], item["options"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            
            out_text = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_letter = extract_option_letter(out_text, item["options"])
            
            # 将结果放入队列
            worker_queue.put({"id": item.get('id'), "pred_letter": pred_letter})
            
    except Exception as e:
        print(f"[ERROR] GPU-{gpu_id} process failed: {e}")

def run_single_agent_evaluation(model_path: Path, data_file: Path, gpu_ids: list, max_new_tokens: int) -> dict:
    """使用单个代理模型对数据进行评估，并返回一个以ID为键的预测结果字典。"""
    print("\n" + "-"*80)
    print(f"🕵️  开始使用代理进行评估: {model_path.name}")
    
    try:
        with data_file.open(encoding="utf8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] 数据文件未找到: {data_file}")
        return {}

    num_gpus = len(gpu_ids)
    data_chunks = np.array_split(test_data, num_gpus)
    data_chunks = [chunk.tolist() for chunk in data_chunks]

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    
    tasks = [(gpu_ids[i], data_chunks[i], model_path, max_new_tokens, results_queue) for i in range(num_gpus)]
    
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(processes=num_gpus) as pool:
        pool.starmap(inference_worker, tasks)

    predictions = {}
    while not results_queue.empty():
        result = results_queue.get()
        if result['id'] is not None:
            predictions[str(result['id'])] = result
            
    print(f"✅ 代理评估完成，共得到 {len(predictions)} 条预测结果。")
    print("-"*80)
    return predictions



def main():
    """主函数，负责编排双代理验证流程。"""
    parser = argparse.ArgumentParser(description="使用双代理交叉验证蒸馏数据的质量。")
    parser.add_argument("--agent_model_1_path", type=str, required=True, help="第一个评估代理模型的路径。")
    parser.add_argument("--agent_model_2_path", type=str, required=True, help="第二个评估代理模型的路径。")
    parser.add_argument("--input_file", type=str, required=True, help="待验证的蒸馏数据文件路径 (.jsonl)。")
    parser.add_argument("--output_file", type=str, required=True, help="保存通过验证的数据的输出文件路径 (.jsonl)。")
    parser.add_argument("--gpu_ids", type=str, required=True, help="用于推理的GPU ID列表，以逗号分隔（例如 '0,1,2'）。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="模型生成新token的最大数量。")

    args = parser.parse_args()

    # --- 解析参数 ---
    try:
        gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
        if not gpu_ids: raise ValueError
    except (ValueError, IndexError):
        print("错误: 'gpu_ids' 参数格式不正确。请提供一个逗号分隔的数字列表，例如 '0,1,2'。")
        return

    agent_1_path = Path(args.agent_model_1_path)
    agent_2_path = Path(args.agent_model_2_path)
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🚀 开始蒸馏数据双代理验证流程")
    print(f"   - 输入文件: {input_file}")
    print(f"   - 输出文件: {output_file}")
    print("="*80)

    # --- 步骤1: 使用代理1进行评估 ---
    preds_agent_1 = run_single_agent_evaluation(agent_1_path, input_file, gpu_ids, args.max_new_tokens)

    # --- 步骤2: 使用代理2进行评估 ---
    preds_agent_2 = run_single_agent_evaluation(agent_2_path, input_file, gpu_ids, args.max_new_tokens)

    # --- 步骤3: 交叉验证并写入最终结果 ---
    print("\n🔍 开始交叉验证...")
    
    validated_data = []
    total_count = 0
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            original_item = json.loads(line)
            total_count += 1
            item_id = str(original_item.get('id'))
            
            # 获取两个代理的预测结果
            pred_1 = preds_agent_1.get(item_id, {}).get('pred_letter')
            pred_2 = preds_agent_2.get(item_id, {}).get('pred_letter')
            
            # 获取标准答案
            ground_truth = original_item.get('answer_idx')
            
            # 只有当两个代理都预测正确时，才认为数据有效
            if ground_truth and pred_1 == ground_truth and pred_2 == ground_truth:
                validated_data.append(original_item)

    # 写入通过验证的数据
    with output_file.open('w', encoding='utf-8') as f:
        for item in validated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    pass_rate = (len(validated_data) / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "="*80)
    print("✅ 验证流程全部完成！")
    print(f"   - 总共处理数据条数: {total_count}")
    print(f"   - 通过双重验证数量: {len(validated_data)}")
    print(f"   - 数据通过率: {pass_rate:.2f}%")
    print(f"   - 高质量蒸馏数据已保存至: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()