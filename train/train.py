import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List
import argparse

# --- 2. 自定义数据集 ---
class DistillationCausalLMDataset(Dataset):
    def __init__(self, tokenizer, file_path, teacher_logits_dir):
        self.tokenizer = tokenizer
        self.data = []
        
        # 在分布式训练中，只有主进程打印加载信息
        is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
        if is_main_process:
            print(f"正在从 {file_path} 将所有数据和logits预加载到内存中...")
        
        temp_metadata = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    temp_metadata.append(json.loads(line))

        # 使用 tqdm 时，通过 disable 参数控制只在主进程显示进度条
        progress_bar = tqdm(temp_metadata, desc="处理数据并加载logits", disable=not is_main_process)
        for entry in progress_bar:
            entry_id = entry.get('id')
            if entry_id is None:
                if is_main_process: print(f"警告: 数据条目缺少 'id'，已跳过: {entry}")
                continue
            
            logits_path = os.path.join(teacher_logits_dir, f"{entry_id}.pt")
            
            if not os.path.exists(logits_path):
                continue
            
            try:
                teacher_logits = torch.load(logits_path, map_location='cpu')
            except Exception as e:
                if is_main_process: print(f"警告: 无法加载logits文件 {logits_path}，已跳过。错误: {e}")
                continue

            question = entry['question']
            options = entry['options']
            options_str = "\\n".join([f"{key}. {value}" for key, value in sorted(options.items())])
            prompt = (
                f"你是一位顶级的医学专家。请仔细分析以下医学多项选择题，并直接选择最正确的选项字母。\\n\\n"
                f"问题: {question}\\n\\n"
                f"选项:\\n{options_str}\\n\\n"
                f"正确答案的选项是："
            )
            answer_letter = entry['answer_idx'].upper()
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_sequence = full_prompt + answer_letter
            
            self.data.append({
                "full_sequence": full_sequence,
                "prompt_length": len(self.tokenizer.encode(full_prompt)),
                "teacher_logits": teacher_logits,
            })
        
        if is_main_process:
            print(f"成功加载 {len(self.data)} 条数据及对应的logits。")
            if len(self.data) == 0:
                print("严重警告: 未加载任何数据。请检查 `teacher_logits_dir` 路径以及.jsonl文件中的 'id' 是否与 .pt 文件名匹配。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_inputs = self.tokenizer(
            item['full_sequence'],
            truncation=True,
            max_length=1024,
            padding=False
        )
        
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "teacher_logits": item['teacher_logits'],
            "prompt_length": item['prompt_length']
        }


@dataclass
class CustomDistillationDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        custom_keys = ["teacher_logits", "prompt_length"]
        standard_features = []
        custom_features_list = []
        for feature in features:
            standard_dict = {}
            custom_dict = {}
            for key, value in feature.items():
                if key in custom_keys:
                    custom_dict[key] = value
                else:
                    standard_dict[key] = value
            standard_features.append(standard_dict)
            custom_features_list.append(custom_dict)
        batch = self.tokenizer.pad(
            standard_features, padding=True, return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100
        teacher_logits = [d["teacher_logits"] for d in custom_features_list]
        prompt_lengths = [d["prompt_length"] for d in custom_features_list]
        batch["teacher_logits"] = torch.stack(teacher_logits)
        batch["prompt_length"] = torch.tensor(prompt_lengths, dtype=torch.long)
        return batch


class DistillationTrainerForCausalLM(Trainer):
    def __init__(self, *args, temperature=2.0, alpha_ce=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.option_keys = ['A', 'B', 'C', 'D', 'E']
        option_tokens = [" " + key for key in self.option_keys]
        self.option_token_ids_map = {
            key: self.tokenizer(token, add_special_tokens=False).input_ids[0] 
            for key, token in zip(self.option_keys, option_tokens)
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits = inputs.pop("teacher_logits").to(self.args.device)
        prompt_lengths = inputs.pop("prompt_length")
        
        num_options = teacher_logits.shape[-1]
        teacher_option_keys = self.option_keys[:num_options]
        teacher_option_token_ids = [self.option_token_ids_map[key] for key in teacher_option_keys]

        outputs = model(**inputs)
        loss_ce = outputs.loss
        student_logits = outputs.logits
        batch_indices = torch.arange(student_logits.size(0), device=student_logits.device)
        answer_token_logits = student_logits[batch_indices, prompt_lengths - 1, :]
        
        student_option_logits = answer_token_logits[:, teacher_option_token_ids]
        
        loss_distill = F.kl_div(
            input=F.log_softmax(student_option_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        loss = self.alpha_ce * loss_distill + (1.0 - self.alpha_ce) * loss_ce
        return (loss, outputs) if return_outputs else loss

def run_training(args):
    """封装核心训练逻辑的函数。"""
    # accelerate 会自动处理设备分配，这里不需要手动设置
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if student_model.config.pad_token_id is None:
        student_model.config.pad_token_id = tokenizer.eos_token_id

    train_dataset = DistillationCausalLMDataset(
        tokenizer=tokenizer,
        file_path=args.train_file_path,
        teacher_logits_dir=args.teacher_logits_dir,
    )
    
    if len(train_dataset) == 0 and os.environ.get("LOCAL_RANK", "0") == "0":
        print("错误: 训练数据集为空，无法开始训练。请检查上游数据生成和验证步骤。")
        return

    data_collator = CustomDistillationDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, 
        bf16=True,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False, 
        optim="paged_adamw_8bit", 
    )

    trainer = DistillationTrainerForCausalLM(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
    )
    
    # 只有主进程打印启动信息
    if trainer.is_world_process_zero():
        print("\n🚀 开始知识蒸馏分布式训练...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 只有主进程负责保存最终模型
    if trainer.is_world_process_zero():
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"\n✅ 训练完成！最终模型已由主进程保存至: {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description="知识蒸馏分布式训练脚本。")
    parser.add_argument("--student_model_path", type=str, required=True)
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--teacher_logits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    args = parser.parse_args()
    
    # 检查是否存在检查点
    latest_checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
    args.resume_from_checkpoint = latest_checkpoint

    # 只有主进程打印配置信息
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("="*80)
        print("📋 训练配置:")
        for arg, value in vars(args).items():
            print(f"   - {arg}: {value}")
        print("="*80)
    
    run_training(args)

if __name__ == "__main__":
    main()