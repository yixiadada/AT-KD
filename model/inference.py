

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


def build_qwen2_prompt(tokenizer, question: str, options: dict):
    """构建与训练脚本一致的Prompt。"""
    options_str = "\n".join([f"{key}. {value}" for key, value in sorted(options.items())])
    prompt_text = (
        f"你是一位顶级的医学专家。请仔细分析以下医学多项选择题，并直接选择最正确的选项字母。\n\n"
        f"问题: {question}\n\n"
        f"选项:\n{options_str}\n\n"
        f"正确答案的选项是："
    )
    messages = [{"role": "user", "content": prompt_text}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return full_prompt

def extract_option_letter(reply: str, options: dict):
    """从模型回复中提取选项字母 (A, B, C, D, E...)"""
    r = reply.strip().upper()
    for k in options:
        if r.startswith(k):
            return k
    for k in options:
        if k in r:
            return k
    return ""

def load_model_for_worker(model_dir: Path, device: str):
    """为单个工作进程加载模型到指定的设备。"""
    print(f"[INFO] Process-{os.getpid()} is loading model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# --- 核心工作进程函数 ---
def inference_worker(gpu_id: int, data_chunk: list, model_path: Path, all_gpu_ids: list, max_new_tokens: int):
    """
    在单个GPU上运行推理的工作函数。
    """
    device = f"cuda:{gpu_id}"
    
    # 将ID最小的GPU进程作为调试输出进程
    is_debug_process = (gpu_id == min(all_gpu_ids))
    
    try:
        tokenizer, model = load_model_for_worker(model_path, device)
        
        outputs, answers, records = [], [], []
        
        progress_bar = tqdm(
            data_chunk, 
            desc=f"GPU-{gpu_id} 推理中", 
            position=gpu_id, 
            unit="题"
        )
        
        for local_idx, item in enumerate(progress_bar):
            prompt = build_qwen2_prompt(tokenizer, item["question"], item["options"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            out_text = tokenizer.decode(
                generated_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # 新增：确保 item 中有 'id' 字段，用于后续匹配
            item_id = item.get('id', f'item_{local_idx}')
            answer_letter = item["answer_idx"].strip().upper()
            pred_letter = extract_option_letter(out_text, item["options"])

            if is_debug_process and local_idx % 20 == 0: # 每20个问题打印一次调试信息
                print(f"\n----- [GPU {gpu_id}] 题目 ID:{item_id} -----")
                print(f"模型原始回复: '{out_text.strip()}'")
                print(f"提取出的选项: '{pred_letter}'")
                print(f"正确答案选项: '{answer_letter}'")
                print(f"结果: {'正确' if pred_letter == answer_letter else '错误'}")
                print("-" * (30 + len(str(item_id))))

            record = {
                "id": item_id, # 确保ID被记录
                "question": item["question"],
                "options": item["options"],
                "answer_letter": answer_letter,
                "answer_text": item["options"].get(answer_letter, "N/A"),
                "model_reply": out_text.strip(),
                "pred_letter": pred_letter
            }
            
            outputs.append(pred_letter)
            answers.append(answer_letter)
            records.append(record)
            
        return outputs, answers, records
    except Exception as e:
        print(f"[ERROR] GPU-{gpu_id} process failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

# --- 任务编排函数 ---
def run_evaluation(model_path: Path, test_file_path: Path, output_path: Path, gpu_ids: list, max_new_tokens: int):
    """对单个数据集执行完整的评测流程。"""
    print("\n" + "="*80)
    print(f"🚀 开始模型评估任务")
    print(f"   - 模型路径: {model_path}")
    print(f"   - 数据文件: {test_file_path}")
    print(f"   - 输出文件: {output_path}")
    print(f"   - 使用 GPUs: {gpu_ids}")
    print("="*80 + "\n")

    start_time = time.time()
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path = output_path.with_suffix('.preds.txt') # 预测选项的附属文件

    # 1. 加载并切分数据
    print(f"[INFO] 正在加载和切分数据...")
    try:
        with test_file_path.open(encoding="utf8") as f:
            test_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"[ERROR] 数据文件未找到: {test_file_path}")
        return

    num_gpus = len(gpu_ids)
    data_chunks = np.array_split(test_data, num_gpus)
    data_chunks = [chunk.tolist() for chunk in data_chunks]

    print(f"[INFO] 数据已切分为 {num_gpus} 块，开始并行推理...")

    # 2. 创建并运行多进程
    tasks = [(gpu_ids[i], data_chunks[i], model_path, gpu_ids, max_new_tokens) for i in range(num_gpus)]
    
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.starmap(inference_worker, tasks)

    # 3. 汇总结果
    print(f"\n[INFO] 正在汇总所有进程的结果...")
    all_outputs, all_answers, all_records = [], [], []
    for outputs, answers, records in results:
        all_outputs.extend(outputs)
        all_answers.extend(answers)
        all_records.extend(records)

    # 4. 写入文件和统计
    with output_path.open("w", encoding="utf8") as f_jsonl:
        for record in all_records:
            json.dump(record, f_jsonl, ensure_ascii=False)
            f_jsonl.write("\n")
            
    with preds_path.open("w", encoding="utf8") as f_preds:
        for pred in all_outputs:
            f_preds.write(pred + "\n")

    total_q = len(all_outputs)
    correct_q = sum(p == a for p, a in zip(all_outputs, all_answers))
    acc = correct_q / total_q if total_q else 0.0

    stats = f"\n总题数: {total_q}\n答对题数: {correct_q}\n准确率: {acc:.2%}"
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print(f"评估完成！")
    print(stats)
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    with output_path.open("a", encoding="utf8") as f:
        f.write(stats)
        f.write(f"\n总耗时: {elapsed_time:.2f} 秒")

    print(f"\n评估结果已保存:")
    print(f"  - 详细回复文件: {output_path}")
    print(f"  - 预测选项文件: {preds_path}")
    print("="*80)


def main():
    """主函数，解析命令行参数并启动评估流程。"""
    parser = argparse.ArgumentParser(description="在多个GPU上并行运行语言模型评估。")
    parser.add_argument("--model_path", type=str, required=True, help="要评估的模型的路径。")
    parser.add_argument("--dataset_path", type=str, required=True, help="评估数据集（.jsonl格式）的路径。")
    parser.add_argument("--output_path", type=str, required=True, help="保存详细评估结果（.jsonl格式）的输出文件路径。")
    parser.add_argument("--gpu_ids", type=str, required=True, help="用于推理的GPU ID列表，以逗号分隔（例如 '0,1,2'）。")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="模型生成新token的最大数量。")
    
    args = parser.parse_args()

    # 解析GPU ID列表
    try:
        gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
        if not gpu_ids:
            raise ValueError
    except (ValueError, IndexError):
        print("错误: 'gpu_ids' 参数格式不正确。请提供一个逗号分隔的数字列表，例如 '0,1,2'。")
        return

    # 启动评估
    run_evaluation(
        model_path=Path(args.model_path),
        test_file_path=Path(args.dataset_path),
        output_path=Path(args.output_path),
        gpu_ids=gpu_ids,
        max_new_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()