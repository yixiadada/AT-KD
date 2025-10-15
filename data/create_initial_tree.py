
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm




MODEL_PATH = ""

# 2. 指定要生成的领域
DOMAIN = "临床医学"


TREE_STRUCTURE = {
    "level_1_nodes": 8,  # 第一层：顶层学科分类
    "level_2_nodes": 5,  # 第二层：每个一级学科下的亚专业
    "level_3_nodes": 5   # 第三层：每个亚专业下的具体知识点/疾病
}

# 4. 指定输出文件的路径
OUTPUT_DIR = ""
OUTPUT_FILENAME = ""

# 5. 配置GPU
GPU_ID = 0

# =========================================================


class AbilityTreeGenerator:
    """
    使用大语言模型生成并构建一个领域能力树。
    """
    def __init__(self, model_path, gpu_id):
        print(f"正在加载模型: {model_path} 到 GPU: {gpu_id}...")
        self.device = f"cuda:{gpu_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self.device
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        print("模型加载完成。")

    def _query_llm(self, prompt, max_new_tokens=512):
        """
        向LLM发送请求并获取回复。
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        outputs = self.pipe(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        response = outputs[0]['generated_text']
        # 提取模型在模板后的实际生成内容
        clean_response = response.split(prompt_text)[-1].strip()
        return clean_response

    def _parse_list_from_response(self, response: str) -> list:
        """从模型的文本回复中解析出列表项。"""
        items = []
        for line in response.split('\n'):
            # 移除前面的数字、点和空格，例如 "1. 心血管内科" -> "心血管内科"
            clean_line = line.strip()
            if '.' in clean_line:
                clean_line = clean_line.split('.', 1)[-1].strip()
            if clean_line:
                items.append(clean_line)
        return items

    def generate_tree(self, domain, structure):
        """
        分层、迭代地生成整个能力树。
        """
        final_tree = {}
        
        # --- 步骤 1: 生成第一层节点 ---
        print(f"\n正在为领域 '{domain}' 生成第一层节点...")
        prompt_l1 = (
            f"你是一位资深的{domain}领域专家和教育家。请为'{domain}'这个领域列出 "
            f"{structure['level_1_nodes']} 个最核心、最重要的一级学科分类。请确保分类之间尽可能独立，"
            "并且覆盖范围广。请直接以编号列表的形式给出，不要包含任何额外的解释。"
        )
        response_l1 = self._query_llm(prompt_l1)
        level1_nodes = self._parse_list_from_response(response_l1)[:structure['level_1_nodes']]
        print(f"生成的第一层节点: {level1_nodes}")

        # --- 步骤 2 & 3: 迭代生成第二层和第三层节点 ---
        pbar_l1 = tqdm(level1_nodes, desc="处理一级学科")
        for l1_node in pbar_l1:
            pbar_l1.set_description(f"处理一级学科: {l1_node}")
            final_tree[l1_node] = {}
            
            prompt_l2 = (
                f"在{domain}领域中，针对一级学科'{l1_node}'，请列出 "
                f"{structure['level_2_nodes']} 个最关键的二级亚专业或子领域。"
                "请直接以编号列表的形式给出，不要包含任何额外的解释。"
            )
            response_l2 = self._query_llm(prompt_l2)
            level2_nodes = self._parse_list_from_response(response_l2)[:structure['level_2_nodes']]
            
            pbar_l2 = tqdm(level2_nodes, desc=f"  处理二级学科", leave=False)
            for l2_node in pbar_l2:
                pbar_l2.set_description(f"  处理二级学科: {l2_node}")
                final_tree[l1_node][l2_node] = {}

                prompt_l3 = (
                    f"在{domain}的'{l1_node} - {l2_node}'亚专业下，请列出 "
                    f"{structure['level_3_nodes']} 个最常见或最重要的具体疾病、"
                    "技术或核心知识点。请直接以编号列表的形式给出，不要包含任何额外的解释。"
                )
                response_l3 = self._query_llm(prompt_l3)
                level3_nodes = self._parse_list_from_response(response_l3)[:structure['level_3_nodes']]
                
                # 为叶子节点设置初始分数为0
                for l3_node in level3_nodes:
                    final_tree[l1_node][l2_node][l3_node] = 0
        
        return final_tree


def main():
    """
    主执行函数。
    """
    print("--- 开始生成初始能力树 ---")
    
    # 初始化生成器
    generator = AbilityTreeGenerator(model_path=MODEL_PATH, gpu_id=GPU_ID)
    
    # 生成树结构
    ability_tree = generator.generate_tree(DOMAIN, TREE_STRUCTURE)
    
    # 保存文件
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n--- 生成完成，正在将能力树保存到: {output_path} ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ability_tree, f, ensure_ascii=False, indent=4)
        print("✅ 能力树保存成功！")
        print("\n最终生成的能力树结构预览:")
        print(json.dumps(ability_tree, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

if __name__ == "__main__":
    main()