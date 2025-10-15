

import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent



# 1. 模型路径配置
TEACHER_MODEL_PATH = ""  
STUDENT_MODEL_PATH = "" 

# 2. 数据集配置
DATASET_NAME = "train" 

DATASET_FILES_PER_ITERATION = [
    "data/medqa/train_id.jsonl", 
    "data/medmcqa/converted_train_id.jsonl", 
     "data/CMExam/CMExam_train_id.jsonl", 
]


ABILITY_CLASS_FILES_PER_ITERATION = [
    "ablity_tree/data/medqa/medqa_train_tree_sorted.jsonl", 
    "ablity_tree/data/medmcqa/medmcqa_train_tree.jsonl", 
    
]

NUM_ITERATIONS = 2  
GPUS_TO_USE = [0, 1, 2] 
NUM_DISTILL_SAMPLES = 10000 

# 4. 初始能力树配置
EXISTING_ABILITY_TREE_PATH = ""

# 5. 训练超参数 (可在此处统一管理)

TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-6
BATCH_SIZE = 2 
GRADIENT_ACCUMULATION_STEPS = 16 

# 6. 最终产出
WORKFLOW_OUTPUT_DIR = f""
MODELS_OUTPUT_DIR = f""


# --- 配置结束 ---
# ==============================================================================


class WorkflowOrchestrator:
    """
    管理和执行整个知识蒸馏工作流的类。
    """
    def __init__(self, config):
        self.config = config
        self.current_student_path = Path(config["student_model_path"])
        self.iteration_states = {} # 存储每一轮的产出文件路径
        self.script_dir = SCRIPT_DIR
        
        # <--- MODIFIED: 创建两个独立的输出目录
        self.workdir = Path(config["workflow_output_dir"])
        self.models_dir = Path(config["models_output_dir"])
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"工作目录 (日志/中间数据) 已创建: {self.workdir.resolve()}")
        print(f"模型目录 (Checkpoints/最终模型) 已创建: {self.models_dir.resolve()}")

    def _run_command(self, step_name: str, command: list, output_paths: list = None):
        """
        通用命令执行函数, 新增检查点功能。
        output_paths: 一个包含此步骤预期产出文件/目录的列表。
        """
        # --- 新增：检查点功能 ---
        if output_paths:
            # Path.exists() 对文件和目录都有效
            all_outputs_exist = all(Path(p).exists() for p in output_paths)
            if all_outputs_exist:
                print("\n" + "="*80)
                print(f"✅ [步骤: {step_name}] 检测到所有输出文件已存在，跳过此步骤。")
                print(f"   - 已检查的路径: {', '.join(map(str, output_paths))}")
                print("="*80)
                return  # 跳过执行

        print("\n" + "="*80)
        print(f"🚀 [步骤: {step_name}] 开始执行...")
        print(f"   - 命令: {' '.join(map(str, command))}")
        
        log_filename = step_name.replace(':', '').replace(' ', '_').replace('-', '_')
        log_path = self.workdir / f"{log_filename}.log"
        print(f"   - 日志文件: {log_path}")


        my_env = os.environ.copy()
        if "accelerate" in command:
            my_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config['gpus']))

        try:
            # 注意: 此处添加了 env=my_env
            with open(log_path, 'w', encoding='utf-8') as log_file:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1, env=my_env)
                for line in iter(process.stdout.readline, ''):
                    print(f"   | {line.strip()}")
                    log_file.write(line)
                process.wait()
                rc = process.poll()

                if rc == 0:
                    print(f"✅ [步骤: {step_name}] 执行成功！")
                else:
                    print(f"❌ [步骤: {step_name}] 执行失败！返回码: {rc}。详情请查看日志: {log_path}")
                    raise RuntimeError(f"Step '{step_name}' failed.")
        except Exception as e:
            print(f"💥 [步骤: {step_name}] 发生严重错误: {e}")
            raise
    
    def run_step_1_create_initial_tree(self):
        """步骤1: 链接已有的初始能力树"""
        source_tree_path_str = self.config.get("existing_ability_tree_path")
        if not source_tree_path_str:
            raise ValueError("错误: 请在全局配置 'EXISTING_ABILITY_TREE_PATH' 中指定您现有能力树的正确路径。")

        source_tree_path = Path(source_tree_path_str)
        target_tree_path = self.workdir / "initial_ability_tree.json"
        self.iteration_states['initial_tree'] = str(target_tree_path)

        print("\n" + "="*80)
        print("🚀 [步骤: 链接初始能力树] 开始执行...")
        print(f"   - 源文件: {source_tree_path}")
        print(f"   - 目标位置: {target_tree_path}")

        if target_tree_path.exists() or target_tree_path.is_symlink():
            print(f"✅ [步骤: 链接初始能力树] 目标文件已存在，跳过链接。")
            return

        if not source_tree_path.is_file():
            source_tree_path = self.script_dir / source_tree_path
            if not source_tree_path.is_file():
                 raise FileNotFoundError(f"错误: 指定的初始能力树文件不存在: {self.config.get('existing_ability_tree_path')}")
        try:
            os.link(source_tree_path, target_tree_path)
            print(f"✅ [步骤: 链接初始能力树] 成功链接文件！")
        except Exception as e:
            print(f"❌ [步骤: 链接初始能力树] 链接文件时出错: {e}")
            raise

    def run_iteration(self, iter_num: int):
        """执行单次完整的迭代"""
        print("\n" + "#"*100)
        print(f"###   开始第 {iter_num} 轮迭代   ###")
        print("#"*100)
        
        # 确定本轮迭代使用的学生模型路径
        # 如果不是第一轮，则学生模型是上一轮的产出
        if iter_num > 1:
            self.current_student_path = self.models_dir / f"iteration_{iter_num - 1}_student_model" / "final_model"
        
        print(f"本轮迭代使用的学生模型: {self.current_student_path}")
        if not self.current_student_path.exists():
             # 对于第一轮，初始学生模型必须存在
            if iter_num == 1:
                raise FileNotFoundError(f"错误: 初始学生模型在路径 '{self.current_student_path}' 未找到！")
            else:
                raise FileNotFoundError(f"错误: 找不到上一轮 ({iter_num - 1}) 训练产出的学生模型 '{self.current_student_path}'。请检查上一轮是否成功完成。")


        current_dataset_file = self.config['dataset_files'][iter_num - 1]
        current_ability_class_file = self.config['ability_class_files'][iter_num - 1]
        print(f"本轮迭代使用的数据集: {current_dataset_file}")
        
        iter_workdir = self.workdir / f"iteration_{iter_num}"
        iter_workdir.mkdir(exist_ok=True)
        
        gpus_str = ",".join(map(str, self.config['gpus']))
        student_model_name = Path(self.current_student_path).name

        # --- 子步骤 A: 评估当前学生模型 (调用 inference.py) ---
        student_reply_dir = iter_workdir / "A_student_reply"
        student_reply_dir.mkdir(exist_ok=True)
        student_replies_file = student_reply_dir / f"{student_model_name}_iter{iter_num}_{self.config['dataset_name']}_replies.jsonl"
        
        cmd_step_a = [
            "python", str(self.script_dir / "inference.py"),
            "--model_path", str(self.current_student_path),
            "--dataset_path", current_dataset_file,
            "--output_path", str(student_replies_file),
            "--gpu_ids", gpus_str
        ]
        self._run_command(f"Iter {iter_num}: Step A - Evaluate Student", cmd_step_a, output_paths=[student_replies_file])

        # --- 子步骤 B: 分析回答，生成学生能力树 (调用 student_data_analyzer.py) ---
        student_tree_dir = iter_workdir / "B_student_tree"
        student_tree_dir.mkdir(exist_ok=True)
        student_ability_tree_file = student_tree_dir / f"{student_model_name}_iter{iter_num}_{self.config['dataset_name']}_tree.json"
        student_stats_file = student_tree_dir / f"{student_model_name}_iter{iter_num}_{self.config['dataset_name']}_stats.json"
        
        cmd_step_b = [
            "python", str(self.script_dir / "student_data_analyzer.py"),
            "--initial_tree_path", self.iteration_states['initial_tree'],
            "--paths_file", current_ability_class_file,
            "--answers_file", current_dataset_file,
            "--student_responses_file", str(student_replies_file),
            "--output_tree_path", str(student_ability_tree_file),
            "--output_stats_path", str(student_stats_file)
        ]
        self._run_command(f"Iter {iter_num}: Step B - Analyze Ability", cmd_step_b, output_paths=[student_ability_tree_file, student_stats_file])

        # --- 子步骤 C.1: 生成蒸馏数据 (调用 generate_distill_data.py) ---
        distill_data_dir = iter_workdir / "C_distill_data"
        distill_data_dir.mkdir(exist_ok=True)
        generated_data_file = distill_data_dir / "generated_data.jsonl"
        
        cmd_step_c1 = [
            "python", str(self.script_dir / "generate_distill_data.py"),
            "--teacher_model_path", self.config['teacher_model_path'],
            "--ability_tree_path", str(student_ability_tree_file),
            "--full_dataset_path", current_dataset_file,
            "--ability_class_path", current_ability_class_file,
            "--output_path", str(generated_data_file),
            "--num_samples", str(self.config['num_distill_samples']),
            "--gpu_ids", gpus_str
        ]
        self._run_command(f"Iter {iter_num}: Step C.1 - Generate Data", cmd_step_c1, output_paths=[generated_data_file])

        # --- 子步骤 C.2: 为蒸馏数据生成教师 Logits (调用 generate_logits.py) ---
        logits_main_dir = distill_data_dir / "teacher_logits_and_report"
        teacher_logits_dir = logits_main_dir / "logits_pt"
        
        cmd_step_c2 = [
            "python", str(self.script_dir / "generate_logits.py"),
            "--model_path", self.config['teacher_model_path'],
            "--dataset_path", str(generated_data_file),
            "--output_dir", str(logits_main_dir),
            "--gpu_ids", gpus_str
        ]
        self._run_command(f"Iter {iter_num}: Step C.2 - Generate Logits", cmd_step_c2, output_paths=[teacher_logits_dir])

        # --- 子步骤 D: 验证蒸馏数据 (调用 evaluate_distilled_data.py) ---
        validated_data_dir = iter_workdir / "D_validated_data"
        validated_data_dir.mkdir(exist_ok=True)
        validated_data_file = validated_data_dir / "validated_data.jsonl"
        
        cmd_step_d = [
            "python", str(self.script_dir / "evaluate_distilled_data.py"),
            "--agent_model_1_path", self.config['teacher_model_path'],
            "--agent_model_2_path", self.config['teacher_model_path'],
            "--input_file", str(generated_data_file),
            "--output_file", str(validated_data_file),
            "--gpu_ids", gpus_str
        ]
        self._run_command(f"Iter {iter_num}: Step D - Validate Data", cmd_step_d, output_paths=[validated_data_file])

        # --- 子步骤 E: 训练学生模型 (调用 train.py) ---
        new_student_model_dir = self.models_dir / f"iteration_{iter_num}_student_model"
        final_model_output_path = new_student_model_dir / "final_model"
        
        num_gpus = len(self.config['gpus'])
        cmd_step_e = [
            "accelerate", "launch",
            f"--num_processes={num_gpus}", # 指定进程数=GPU数
            f"--main_process_port={29500 + iter_num}", # 为每次迭代分配不同端口，避免冲突
            str(self.script_dir / "train.py"),
            "--student_model_path", str(self.current_student_path),
            "--train_file_path", str(validated_data_file),
            "--teacher_logits_dir", str(teacher_logits_dir),
            "--output_dir", str(new_student_model_dir),
            "--num_epochs", str(self.config['train_epochs']),
            "--learning_rate", str(self.config['learning_rate']),
            "--batch_size", str(self.config['batch_size']),
            "--gradient_accumulation_steps", str(self.config['gradient_accumulation_steps']),
        ]
        self._run_command(f"Iter {iter_num}: Step E - Train Student", cmd_step_e, output_paths=[final_model_output_path])
        
        # 更新当前学生模型路径以备下一轮迭代
        self.current_student_path = final_model_output_path
        
        print(f"🎉 第 {iter_num} 轮迭代完成！新的学生模型位于: {self.current_student_path}")

    def run(self):
        """启动整个工作流"""
        print("====== 启动 AT-KD 自动化工作流 ======")
        
        # 1. 链接或验证初始能力树
        self.run_step_1_create_initial_tree()
        
        # 2. 开始迭代
        for i in range(1, self.config["num_iterations"] + 1):
            self.run_iteration(i)

        print("\n\n" + "#"*100)
        print("###   AT-KD 自动化工作流全部执行完毕   ###")
        print(f"###   最终日志和数据保存在: {self.workdir.resolve()}   ###")
        print(f"###   最终学生模型保存在: {self.models_dir.resolve()}   ###")
        print("#"*100)


if __name__ == "__main__":
    # 检查配置是否匹配
    if len(DATASET_FILES_PER_ITERATION) != NUM_ITERATIONS or len(ABILITY_CLASS_FILES_PER_ITERATION) != NUM_ITERATIONS:
        raise ValueError(
            f"配置错误: 'DATASET_FILES_PER_ITERATION' 和 'ABILITY_CLASS_FILES_PER_ITERATION' 列表的长度 "
            f"({len(DATASET_FILES_PER_ITERATION)}, {len(ABILITY_CLASS_FILES_PER_ITERATION)}) "
            f"必须与 'NUM_ITERATIONS' ({NUM_ITERATIONS}) 相等。"
        )

    # 将配置字典化
    main_config = {
        "teacher_model_path": TEACHER_MODEL_PATH,
        "student_model_path": STUDENT_MODEL_PATH,
        "dataset_name": DATASET_NAME,
        "dataset_files": DATASET_FILES_PER_ITERATION,
        "ability_class_files": ABILITY_CLASS_FILES_PER_ITERATION,
        "num_iterations": NUM_ITERATIONS,
        "gpus": GPUS_TO_USE,
        "workflow_output_dir": WORKFLOW_OUTPUT_DIR,
        "models_output_dir": MODELS_OUTPUT_DIR,
        "existing_ability_tree_path": EXISTING_ABILITY_TREE_PATH,
        "num_distill_samples": NUM_DISTILL_SAMPLES,
        "train_epochs": TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    }
    
    orchestrator = WorkflowOrchestrator(main_config)
    orchestrator.run()