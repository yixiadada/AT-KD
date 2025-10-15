# AT-KD


## 📋 项目概述

AT-KD 是一个创新的知识蒸馏框架,通过构建和维护领域能力树来指导蒸馏过程。框架能够:

- 🎯 **自适应数据生成**: 根据学生模型的弱项动态生成针对性训练数据
- 🌳 **能力树管理**: 构建和更新多层级领域知识能力树
- 🔄 **迭代优化**: 支持多轮迭代训练,持续提升模型能力
- ✅ **质量保证**: 双代理交叉验证机制确保蒸馏数据质量
- 🚀 **分布式支持**: 全流程支持多GPU并行加速


## 🔧 环境要求

### 基础环境
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### 依赖安装

```bash
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install tqdm
pip install numpy
```

## 🚀 快速开始

### 1. 准备初始能力树

首先需要为你的领域创建一个初始能力树:

```bash
python create_initial_tree.py
```

**配置 `create_initial_tree.py`**:
```python
MODEL_PATH = "/path/to/your/llm"  # 用于生成能力树的大模型
DOMAIN = "临床医学"  # 你的目标领域
OUTPUT_DIR = "./ability_trees"
OUTPUT_FILENAME = "initial_tree.json"
GPU_ID = 0
```

### 2. 准备数据文件

确保你有以下格式的数据文件:

**训练数据格式** (`.jsonl`):
```json
{
  "id": 0,
  "question": "患者出现胸痛...",
  "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
  "answer_idx": "C",
  "meta_info": "额外信息"
}
```

**能力分类文件** (`.jsonl`):
```json
{
  "id": 0,
  "ability_path": ["临床医学", "内科学", "心血管疾病"]
}
```

### 3. 配置主工作流

编辑 `main.py` 中的配置部分:

```python
# 1. 模型路径配置
TEACHER_MODEL_PATH = "/path/to/teacher/model"  
STUDENT_MODEL_PATH = "/path/to/student/model" 

# 2. 数据集配置
DATASET_FILES_PER_ITERATION = [
    "data/iteration1/train.jsonl",
    "data/iteration2/train.jsonl",
]

ABILITY_CLASS_FILES_PER_ITERATION = [
    "data/iteration1/ability_paths.jsonl",
    "data/iteration2/ability_paths.jsonl",
]

NUM_ITERATIONS = 2  # 迭代轮数
GPUS_TO_USE = [0, 1, 2]  # 使用的GPU ID
NUM_DISTILL_SAMPLES = 10000  # 每轮生成的蒸馏数据量

# 4. 初始能力树配置
EXISTING_ABILITY_TREE_PATH = "ability_trees/initial_tree.json"

# 5. 训练超参数
TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16

# 6. 输出目录
WORKFLOW_OUTPUT_DIR = "./output/workflow"
MODELS_OUTPUT_DIR = "./output/models"
```

### 4. 启动训练

```bash
python main.py
```

## 📁 文件说明

### 核心脚本

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `main.py` | 工作流编排器 | 配置参数 | 完整训练流程 |
| `create_initial_tree.py` | 创建初始能力树 | 领域名称 | 能力树JSON |
| `inference.py` | 模型推理评估 | 模型+数据集 | 评估结果 |
| `student_data_analyzer.py` | 能力分析 | 学生回答+标准答案 | 更新的能力树 |
| `generate_distill_data.py` | 生成蒸馏数据 | 能力树+种子数据 | 蒸馏数据 |
| `generate_logits.py` | 生成教师logits | 教师模型+数据 | Logits文件 |
| `evaluate_distilled_data.py` | 双代理验证 | 两个代理模型+数据 | 验证后数据 |
| `train.py` | 知识蒸馏训练 | 学生模型+蒸馏数据+logits | 训练后模型 |

### 输出结构

```
output/
├── workflow/                    # 工作流中间数据和日志
│   ├── initial_ability_tree.json
│   ├── iteration_1/
│   │   ├── A_student_reply/    # 学生回答
│   │   ├── B_student_tree/     # 能力分析
│   │   ├── C_distill_data/     # 蒸馏数据
│   │   ├── D_validated_data/   # 验证后数据
│   │   └── *.log               # 各步骤日志
│   └── iteration_2/
│       └── ...
│
└── models/                      # 训练后的模型
    ├── iteration_1_student_model/
    │   ├── checkpoint-1000/
    │   └── final_model/
    └── iteration_2_student_model/
        └── final_model/
```

## ⚙️ 高级配置

### 断点续训

框架支持自动断点续训:
- 检测到已存在的输出文件会自动跳过该步骤
- 训练脚本自动识别最新checkpoint并继续训练

### 分布式训练配置

训练脚本使用 Accelerate 进行分布式训练:

```bash
# main.py 会自动调用 accelerate launch
# 你只需在 GPUS_TO_USE 中指定GPU即可
GPUS_TO_USE = [0, 1, 2, 3]  # 使用4个GPU
```

### 自定义蒸馏参数

在 `train.py` 中调整蒸馏超参数:

```python
--temperature 2.0           # 蒸馏温度
--alpha_ce 0.5             # CE loss权重 (1-alpha为KD loss权重)
```




---

**开始你的知识蒸馏之旅吧! 🚀**
