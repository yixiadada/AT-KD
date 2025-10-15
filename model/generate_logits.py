# 文件名: generate_logits.py
# 描述: (已参数化改造) 扮演“教师 Logits 生成代理”的角色。
#       读取蒸馏数据，使用教师模型为其生成答案选项的 logits，
#       并将每个数据点的 logits 保存为独立的 .pt 文件，以供训练脚本使用。
#       同时，此脚本也会评估教师模型在这些生成数据上的准确率。

import torch
import torch.multiprocessing as mp
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path

# =============================================================================
# --- 核心功能函数 ---
# =============================================================================

def get_option_logits_from_teacher(model, tokenizer, data_entry):
    """
    为单个条目提取固定选项（A-E）的logits，并根据实际可用选项判断预测结果。
    返回: (fixed_option_logits, predicted_letter)
           其中 fixed_option_logits 是一个形状固定为 [5] 的张量。
    """
    TARGET_KEYS = ['A', 'B', 'C', 'D', 'E']
    
    question = data_entry['question']
    options = data_entry['options']
    available_keys = sorted(options.keys())
    
    options_str = "\n".join([f"{key}. {value}" for key, value in sorted(options.items())])

    prompt_text = (
        f"你是一位顶级的医学专家。请仔细分析以下医学多项选择题。\n\n"
        f"问题: {question}\n\n"
        f"选项:\n{options_str}\n\n"
        f"正确答案的选项是："
    )

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]

    target_option_tokens = [" " + key for key in TARGET_KEYS]
    try:
        target_token_ids_list = tokenizer(target_option_tokens, add_special_tokens=False).input_ids
        target_token_ids = [ids[0] for ids in target_token_ids_list]
    except (IndexError, TypeError):
        target_token_ids_list = tokenizer(TARGET_KEYS, add_special_tokens=False).input_ids
        target_token_ids = [ids[0] for ids in target_token_ids_list]
        
    fixed_option_logits = next_token_logits[target_token_ids].to(torch.float32)

    mask = torch.full_like(fixed_option_logits, -float('inf'))
    
    for i, key in enumerate(TARGET_KEYS):
        if key in available_keys:
            mask[i] = 0.0
    
    masked_logits = fixed_option_logits + mask
    
    pred_idx = torch.argmax(masked_logits).item()
    pred_letter = TARGET_KEYS[pred_idx]
    
    return fixed_option_logits, pred_letter


def worker(rank, world_size, data_chunks, args, temp_file_pattern, logits_output_dir):
    """每个GPU上运行的工作进程"""
    gpu_id = args.gpu_ids[rank]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f"cuda:0")
    print(f"工作进程 Rank {rank} 已启动，使用 GPU: {gpu_id}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    data_chunk = data_chunks[rank]
    results_metadata = []

    progress_bar = tqdm(data_chunk, total=len(data_chunk), desc=f"GPU {gpu_id}", position=rank)
    
    for entry in progress_bar:
        try:
            entry_id = entry.get('id')
            if entry_id is None:
                print(f"GPU {gpu_id} 发现条目缺少 'id'，已跳过。")
                continue

            option_logits, pred_letter = get_option_logits_from_teacher(model, tokenizer, entry)
            
            pt_output_path = logits_output_dir / f"{entry_id}.pt"
            torch.save(option_logits.cpu(), pt_output_path)
            
            answer_letter = entry['answer_idx'].strip().upper()

            # 元数据用于生成最终的jsonl文件和计算准确率
            result_entry = {
                'id': entry_id,
                'pred_letter': pred_letter,
                'answer_letter': answer_letter,
            }
            results_metadata.append(result_entry)
        except Exception as e:
            print(f"GPU {rank} 在处理 id={entry.get('id', 'N/A')} 时出错: {e}")

    # 将该进程的元数据结果写入临时文件
    temp_file_path = temp_file_pattern.format(rank=rank)
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        for res in results_metadata:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"工作进程 Rank {rank} 已完成。")


def main():
    parser = argparse.ArgumentParser(description="为数据集并行生成教师 Logits 并评估准确率。")
    parser.add_argument('--model_path', type=str, required=True, help="教师模型的路径。")
    parser.add_argument('--dataset_path', type=str, required=True, help="输入数据集的路径 (.jsonl)。")
    parser.add_argument('--output_dir', type=str, required=True, help="保存所有输出的主目录。")
    parser.add_argument('--gpu_ids', type=str, required=True, help="指定使用的GPU设备ID，以逗号分隔 (例如 '0,1')")
    
    args = parser.parse_args()
    
    try:
        args.gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
        world_size = len(args.gpu_ids)
        print(f"将在 {world_size} 个GPU上运行: {args.gpu_ids}")
    except ValueError:
        print("错误: 'gpu_ids' 参数格式不正确。请提供一个逗号分隔的数字列表。")
        return

    # --- 创建输出目录 ---
    main_output_dir = Path(args.output_dir)
    logits_output_dir = main_output_dir / 'logits_pt'
    temp_files_dir = main_output_dir / 'temp_results'
    
    main_output_dir.mkdir(parents=True, exist_ok=True)
    logits_output_dir.mkdir(exist_ok=True)
    temp_files_dir.mkdir(exist_ok=True)

    # --- 加载和分发数据 ---
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    if not dataset:
        print(f"警告: 输入文件 '{args.dataset_path}' 为空或无法解析，流程终止。")
        return

    chunk_size = (len(dataset) + world_size - 1) // world_size
    data_chunks = [dataset[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

    # --- 启动多进程 ---
    temp_file_pattern = str(temp_files_dir / "temp_results_gpu_{rank}.jsonl")
    mp.spawn(worker, args=(world_size, data_chunks, args, temp_file_pattern, logits_output_dir), nprocs=world_size, join=True)

    # --- 合并元数据结果 ---
    print("\n所有进程已完成。正在合并元数据结果...")
    final_results = []
    for i in range(world_size):
        rank = args.gpu_ids[i]
        temp_file = temp_file_pattern.format(rank=rank)
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    final_results.append(json.loads(line))
            os.remove(temp_file) 

    final_results.sort(key=lambda x: x['id'])
    
    print(f"所有独立的 .pt 文件已保存至目录: {logits_output_dir}")

    # --- 计算并保存准确率 ---
    print("\n正在计算教师模型在生成数据上的准确率...")
    if not final_results:
        print("没有可供评估的结果。")
        return

    correct_q = sum(1 for result in final_results if result.get('pred_letter') == result.get('answer_letter'))
    total_q = len(final_results)
    accuracy = correct_q / total_q if total_q > 0 else 0.0

    report_path = main_output_dir / "teacher_accuracy_on_distilled_data_report.txt"
    stats_report = (
        f"===== 教师模型在生成数据上的评估报告 =====\n"
        f"模型: {args.model_path}\n"
        f"评估数据: {args.dataset_path}\n"
        f"总题数: {total_q}\n"
        f"答对题数: {correct_q}\n"
        f"准确率: {accuracy:.2%}\n"
        f"=========================================\n"
    )

    print("\n--- 评估结果 ---")
    print(stats_report)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(stats_report)
    
    print(f"准确率报告已保存至: {report_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()