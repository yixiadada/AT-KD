

import json
import random
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob
import os
import torch.multiprocessing as mp
import re
from tqdm import tqdm
import argparse
import tempfile
from pathlib import Path


def load_local_model(model_path: str, gpu_id: str):
    tqdm.write(f"⏳ [GPU {gpu_id}] 正在从 '{model_path}' 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=f'cuda:{gpu_id}'
        )
        tqdm.write(f"✅ [GPU {gpu_id}] 模型加载成功！")
        return model, tokenizer
    except Exception as e:
        tqdm.write(f"❌ [GPU {gpu_id}] 模型加载失败: {e}"); return None, None

def load_full_dataset(file_path: str):
    dataset_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f: 
            if line.strip():
                data_item = json.loads(line)
                dataset_map[str(data_item['id'])] = data_item
    return dataset_map

def load_ability_classifications(file_path: str):
    ability_to_ids = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ability_to_ids[tuple(item['ability_path'])].append(str(item['id']))
    return ability_to_ids

def load_ability_tree(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)

def get_leaf_abilities(node, path=None):
    if path is None: path = []
    leaf_scores = {}
    if not isinstance(node, dict):
        return {}
    for name, value in node.items():
        if name == '综合得分': continue
        if isinstance(value, dict): 
            leaf_scores.update(get_leaf_abilities(value, path + [name]))
        elif isinstance(value, (int, float)): 
            leaf_scores[tuple(path + [name])] = value
    return leaf_scores


def calculate_generation_allocation(ability_tree, total_count):
    leaf_scores = get_leaf_abilities(ability_tree)
    if not leaf_scores: return {}
    abilities, scores = list(leaf_scores.keys()), np.array(list(leaf_scores.values()))
    

    weights = 1.0 / (scores - np.min(scores) + 1e-6)
    normalized_weights = weights / np.sum(weights)
    
    raw_counts = normalized_weights * total_count
    counts = np.floor(raw_counts).astype(int)
    

    remainder = total_count - np.sum(counts)
    if remainder > 0:
        top_indices = np.argsort(raw_counts - counts)[-int(remainder):]
        counts[top_indices] += 1
        
    return {abilities[i]: counts[i] for i in range(len(abilities)) if counts[i] > 0}

def parse_generated_text_robustly(text_block: str, gpu_id: str):
    try:
        question_match = re.search(r"题干[:：\s]*(.*?)\n", text_block, re.DOTALL)
        if not question_match: return None
        question_text = question_match.group(1).strip()

        options = {}
        option_matches = re.findall(r"([A-Z])\.\s*(.*?)\n", text_block)
        for match in option_matches:
            options[match[0]] = match[1].strip()
        if not options: return None

        answer_match = re.search(r"答案[:：\s]*([A-Z])", text_block)
        answer_idx = answer_match.group(1).strip() if answer_match else "N/A"
        answer_text = options.get(answer_idx, "N/A")

        return {
            "question": question_text, "options": options,
            "answer": answer_text, "answer_idx": answer_idx
        }
    except Exception as e:
        tqdm.write(f"⚠️ [GPU {gpu_id}] 在宽松解析模式下仍然出错: {e}")
        return None

def generate_similar_question(seed_question: dict, model, tokenizer, gpu_id):
    ability_path_str = " -> ".join(seed_question['ability_path'])
    options_str = "\n".join([f"{k}. {v}" for k, v in seed_question['options'].items()])
    prompt = f"""你是一位顶级的医学出题专家。请严格模仿以下“种子问题”的风格、知识点（领域：{ability_path_str}）和格式，创造1道全新的、高质量的单项选择题。\n\n要求：\n1. 新问题的考点、难度和选项数量应与种子问题高度相似。\n2. 必须使用不同的提问方式、案例或情景，严禁抄袭原文。\n3. 严格按照下面的格式输出，不要包含任何额外的解释或文字。\n\n[START]\n题干: [这里是新问题的题干]\nA. [选项A的内容]\nB. [选项B的内容]\nC. [选项C的内容]\nD. [选项D的内容]\nE. [选项E的内容]\n答案: [这里是正确选项的字母，例如：C]\n[END]\n\n---\n种子问题参考：\n题干: {seed_question['question']}\n{options_str}\n答案: {seed_question['answer_idx']}\n---\n\n现在，请开始生成："""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, num_return_sequences=1, do_sample=True, temperature=0.75, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    generation_marker = "现在，请开始生成："
    model_output_only = generated_text.split(generation_marker)[-1]

    try:
        block = model_output_only.split('[START]')[1].split('[END]')[0].strip()
        parsed_data = parse_generated_text_robustly(block, gpu_id)
    except IndexError:
        tqdm.write(f"🟡 [GPU {gpu_id}] ID:{seed_question.get('id', 'N/A')} 未找到[START]/[END]标签，尝试宽松解析...")
        parsed_data = parse_generated_text_robustly(model_output_only, gpu_id)

    if parsed_data:
        parsed_data.update({
            "meta_info": seed_question.get('meta_info', ''),
            "ability_path": seed_question['ability_path'],
            "generated_from_seed": seed_question['id']
        })
        return parsed_data
    else:
        tqdm.write(f"⚠️ [GPU {gpu_id}] ID:{seed_question.get('id', 'N/A')} 所有解析尝试均失败。")
        return None

def generation_worker(rank, world_size, gpus, config):
    gpu_id = gpus[rank]
    tqdm.write(f"--- Worker启动：进程Rank {rank}，使用GPU {gpu_id} ---")

    model, tokenizer = load_local_model(config['teacher_model_path'], str(gpu_id))
    if not model: return

    full_dataset_map = load_full_dataset(config['full_dataset_path'])
    ability_to_ids_map = load_ability_classifications(config['ability_class_path'])
    ability_tree = load_ability_tree(config['ability_tree_path'])
    generation_plan = calculate_generation_allocation(ability_tree, config['num_samples'])
    
    all_tasks = []
    for ability_path, count in generation_plan.items():
        candidate_ids = ability_to_ids_map.get(ability_path)
        if candidate_ids:
            for seed_id in random.choices(candidate_ids, k=count):
                all_tasks.append({'ability_path': ability_path, 'question_data': full_dataset_map.get(seed_id)})
    
    random.seed(42 + rank) # 为每个进程设置不同随机种子
    random.shuffle(all_tasks)
    
    tasks_for_this_rank = all_tasks[rank::world_size]
    tqdm.write(f"--- [GPU {gpu_id}] 分配到 {len(tasks_for_this_rank)} 个生成任务 ---")

    output_file = config['output_file_template'].format(rank=rank)
    with open(output_file, 'w', encoding='utf-8') as f:
        progress_bar = tqdm(tasks_for_this_rank, desc=f"🚀 [GPU {gpu_id}]", position=rank, unit="条")
        for task in progress_bar:
            if task['question_data']:
                task['question_data']['ability_path'] = list(task['ability_path'])
                new_question = None
                for attempt in range(config['max_retries']):
                    new_question = generate_similar_question(task['question_data'], model, tokenizer, gpu_id)
                    if new_question: break
                if new_question:
                    f.write(json.dumps(new_question, ensure_ascii=False) + '\n')

    tqdm.write(f"🎉 [GPU {gpu_id}] 任务完成，临时文件已保存。")

def merge_files(file_pattern, output_file):
    input_files = sorted(glob.glob(file_pattern))
    if not input_files:
        print(f"❌ 合并失败：未找到任何匹配 '{file_pattern}' 的文件。")
        return
    print(f"\n🔍 找到 {len(input_files)} 个临时文件进行合并...")

    total_lines = 0
    current_id = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in input_files:
            with open(filename, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        try:
                            data_item = json.loads(line)
                            data_item['id'] = current_id
                            outfile.write(json.dumps(data_item, ensure_ascii=False) + '\n')
                            total_lines += 1
                            current_id += 1
                        except json.JSONDecodeError:
                            print(f"⚠️ 警告: 在文件 {filename} 中发现无效的JSON行，已跳过。")
    print(f"✅ 合并完成！总共合并了 {total_lines} 行数据到 '{output_file}'。")

# ==============================================================================
# SECTION 2: 主执行流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="根据学生能力树生成针对性蒸馏数据。")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="教师模型的路径。")
    parser.add_argument("--ability_tree_path", type=str, required=True, help="学生能力树文件路径 (.json)。")
    parser.add_argument("--full_dataset_path", type=str, required=True, help="用于选择种子数据的完整数据集路径 (.jsonl)。")
    parser.add_argument("--ability_class_path", type=str, required=True, help="问题ID到能力路径的映射文件路径 (.jsonl)。")
    parser.add_argument("--output_path", type=str, required=True, help="保存生成的蒸馏数据的最终输出文件路径 (.jsonl)。")
    parser.add_argument("--num_samples", type=int, required=True, help="要生成的蒸馏数据总数。")
    parser.add_argument("--gpu_ids", type=str, required=True, help="用于推理的GPU ID列表，以逗号分隔（例如 '0,1,2'）。")
    parser.add_argument("--max_retries", type=int, default=3, help="每个生成任务的最大重试次数。")
    
    args = parser.parse_args()

    try:
        gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
        if not gpu_ids: raise ValueError
        world_size = len(gpu_ids)
    except (ValueError, IndexError):
        print("错误: 'gpu_ids' 参数格式不正确。请提供一个逗号分隔的数字列表，例如 '0,1,2'。")
        return
        
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"🚀 STAGE 1: 开始分布式数据生成...")
    print(f"   将在 {world_size} 个GPU上启动: {gpu_ids}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'num_samples': args.num_samples,
            'teacher_model_path': args.teacher_model_path,
            'full_dataset_path': args.full_dataset_path,
            'ability_tree_path': args.ability_tree_path,
            'ability_class_path': args.ability_class_path,
            'max_retries': args.max_retries,
            'output_file_template': os.path.join(temp_dir, "rank_{rank}.jsonl")
        }
        
        mp.spawn(generation_worker, args=(world_size, gpu_ids, config), nprocs=world_size, join=True)
        
        print("\n" + "="*80)
        print("✅ STAGE 1: 所有生成进程均已完成。")
        print("="*80)

        print(f"\n🚀 STAGE 2: 开始合并临时文件...")
        temp_file_pattern = os.path.join(temp_dir, "rank_*.jsonl")
        merge_files(temp_file_pattern, args.output_path)

    print("\n" + "="*80)
    print("🎉🎉🎉 全部流程执行完毕！ 🎉🎉🎉")
    print(f"最终产出文件位于: {args.output_path}")
    print("="*80)

if __name__ == '__main__':
    main()