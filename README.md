# AT-KD

[中文版](./README_zh.md) | English

## 🔧 Requirements

### Environment
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### Installation

```bash
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install tqdm
pip install numpy
```

## 🚀 Quick Start

### 1. Prepare Initial Ability Tree

First, create an initial ability tree for your domain:

```bash
python create_initial_tree.py
```

**Configure `create_initial_tree.py`**:
```python
MODEL_PATH = "/path/to/your/llm"  # LLM for generating ability tree
DOMAIN = "Clinical Medicine"  # Your target domain
OUTPUT_DIR = "./ability_trees"
OUTPUT_FILENAME = "initial_tree.json"
GPU_ID = 0
```

### 2. Prepare Data Files

Ensure you have data files in the following formats:

**Training Data Format** (`.jsonl`):
```json
{
  "id": 0,
  "question": "A patient presents with chest pain...",
  "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
  "answer_idx": "C",
  "meta_info": "Additional information"
}
```

**Ability Classification File** (`.jsonl`):
```json
{
  "id": 0,
  "ability_path": ["Clinical Medicine", "Internal Medicine", "Cardiovascular Disease"]
}
```

### 3. Configure Main Workflow

Edit the configuration section in `main.py`:

```python
# 1. Model Path Configuration
TEACHER_MODEL_PATH = "/path/to/teacher/model"  
STUDENT_MODEL_PATH = "/path/to/student/model" 

# 2. Dataset Configuration
DATASET_FILES_PER_ITERATION = [
    "data/iteration1/train.jsonl",
    "data/iteration2/train.jsonl",
]

ABILITY_CLASS_FILES_PER_ITERATION = [
    "data/iteration1/ability_paths.jsonl",
    "data/iteration2/ability_paths.jsonl",
]

NUM_ITERATIONS = 2  # Number of iterations
GPUS_TO_USE = [0, 1, 2]  # GPU IDs to use
NUM_DISTILL_SAMPLES = 10000  # Number of distillation samples per iteration

# 4. Initial Ability Tree Configuration
EXISTING_ABILITY_TREE_PATH = "ability_trees/initial_tree.json"

# 5. Training Hyperparameters
TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16

# 6. Output Directories
WORKFLOW_OUTPUT_DIR = "./output/workflow"
MODELS_OUTPUT_DIR = "./output/models"
```

### 4. Launch Training

```bash
python main.py
```



**Start your knowledge distillation journey! 🚀**
