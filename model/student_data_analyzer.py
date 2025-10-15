
import json
import os
from typing import List, Dict, Any
import argparse 

class AbilityEvaluator:

    SCORE_KEY = "综合得分"  # 为类别节点设置的计分键

    def __init__(self, tree_filepath: str):

        print(f"正在从 {tree_filepath} 加载初始能力树...")
        self.ability_tree = self._load_json(tree_filepath)
        if not self.ability_tree:
            raise ValueError("能力树文件加载失败或为空。")
        
        print("正在初始化能力树，为所有类别节点添加“综合得分”项...")
        self._initialize_branch_scores(self.ability_tree)
        
        # 初始化用于存储二级节点统计数据的字典
        self.statistics = {}
        
        print("能力树初始化完成。")

    def _load_json(self, filepath: str) -> Dict:
       
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"错误: 无法加载或解析文件 {filepath}。{e}")
            return {}

    def _load_jsonl_as_map(self, filepath: str) -> Dict[str, Dict]:
       
        data_map = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip(): continue
                    item = json.loads(line)
                    # 确保ID是字符串类型以便匹配
                    item_id = item.get('id')
                    if item_id is None:
                        print(f"警告: 文件 {filepath} 第 {i+1} 行缺少 'id'，已跳过。")
                        continue
                    data_map[str(item_id)] = item
            return data_map
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"错误: 无法加载或解析文件 {filepath}。{e}")
            return {}

    def _initialize_branch_scores(self, node: Dict[str, Any]):
        """递归地为所有分支节点添加计分键。"""
        if self.SCORE_KEY not in node:
            node[self.SCORE_KEY] = 0
        for key, value in node.items():
            if isinstance(value, dict):
                self._initialize_branch_scores(value)

    def evaluate_and_update(self, paths_filepath: str, answers_filepath: str, responses_filepath: str):
        """
        加载数据，比对答案，更新能力树分数，并收集统计数据。
        """
        paths_map = self._load_jsonl_as_map(paths_filepath)
        answers_map = self._load_jsonl_as_map(answers_filepath)
        responses_map = self._load_jsonl_as_map(responses_filepath)

        print(f"\n开始评估 {len(responses_map)} 个学生回答...")
        
        processed_count = 0
        for qid, response_item in responses_map.items():
            qid_str = str(qid)
            if qid_str not in paths_map or qid_str not in answers_map:
                continue

            path_item = paths_map[qid_str]
            answer_item = answers_map[qid_str]
            
            student_answer = response_item.get('pred_letter')
            standard_answer = answer_item.get('answer_idx')
            ability_path = path_item.get('ability_path')

            if student_answer is None or standard_answer is None or not ability_path:
                print(f"警告: 问题ID '{qid_str}' 缺少必要信息，已跳过。")
                continue

            is_correct = str(student_answer).strip() == str(standard_answer).strip()
           
            
            self._update_score_by_path(ability_path, is_correct)
            self._update_statistics(ability_path, is_correct) # 新增：更新统计数据
            processed_count += 1
        
        print(f"评估完成，共处理 {processed_count} 个有效回答。")

    def _update_statistics(self, path: List[str], is_correct: bool):
        """根据能力路径，更新二级节点的统计数据。"""
        if len(path) < 2:
            return
        
        second_level_node = path[1]
        
        if second_level_node not in self.statistics:
            self.statistics[second_level_node] = {
                "total_questions": 0,
                "correct_answers": 0
            }
        
        self.statistics[second_level_node]["total_questions"] += 1
        if is_correct:
            self.statistics[second_level_node]["correct_answers"] += 1

    def _update_score_by_path(self, path: List[str], is_correct: bool):
        """根据路径找到节点并更新其分数。"""
        try:
            current_node = self.ability_tree
            parent_node = None
            last_step = None
            # 更新所有层级的综合得分
            for step in path:
                if self.SCORE_KEY in current_node:
                    current_node[self.SCORE_KEY] += 1 if is_correct else -1
                parent_node = current_node
                current_node = current_node[step]
                last_step = step
            
            # 更新叶子节点的分数
            if isinstance(current_node, (int, float)):
                parent_node[last_step] += 1 if is_correct else -1
            elif isinstance(current_node, dict) and self.SCORE_KEY in current_node:
                # 如果路径的最后一级也是一个分支，更新其综合得分
                current_node[self.SCORE_KEY] += 1 if is_correct else -1
            else:
                 print(f"警告: 路径 '{' -> '.join(path)}' 指向的节点类型未知，无法评分。")
        except (KeyError, TypeError):
            print(f"错误: 无法在能力树中找到或导航路径 '{' -> '.join(path)}'。")

    def save_updated_tree(self, output_path: str):
        """将更新后的能力树保存到文件。"""
        print(f"\n正在将更新后的能力树保存到 {output_path}...")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.ability_tree, f, ensure_ascii=False, indent=4)
            print("能力树保存成功！")
        except Exception as e:
            print(f"错误: 保存能力树文件失败。{e}")

    def calculate_and_save_statistics(self, output_path: str):
        """计算正确率并保存统计结果。"""
        print(f"\n正在计算二级节点正确率并保存到 {output_path}...")
        for node, data in self.statistics.items():
            total = data["total_questions"]
            correct = data["correct_answers"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            data["accuracy"] = f"{accuracy:.2f}%"

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.statistics, f, ensure_ascii=False, indent=4)
            print("统计结果保存成功！")
        except Exception as e:
            print(f"错误: 保存统计文件失败。{e}")


def main():
    """主函数，解析命令行参数并启动评估分析流程。"""
    parser = argparse.ArgumentParser(description="评估学生模型回答并更新能力树。")
    parser.add_argument("--initial_tree_path", type=str, required=True, help="初始能力树模板文件的路径。")
    parser.add_argument("--paths_file", type=str, required=True, help="问题ID到能力路径映射文件的路径 (.jsonl)。")
    parser.add_argument("--answers_file", type=str, required=True, help="标准答案文件的路径 (.jsonl)。")
    parser.add_argument("--student_responses_file", type=str, required=True, help="包含学生模型回答的文件的路径 (.jsonl)。")
    parser.add_argument("--output_tree_path", type=str, required=True, help="保存更新后能力树的输出文件路径 (.json)。")
    parser.add_argument("--output_stats_path", type=str, required=True, help="保存能力统计数据的输出文件路径 (.json)。")
    
    args = parser.parse_args()

    # 检查输入文件是否存在
    required_files = [args.initial_tree_path, args.paths_file, args.answers_file, args.student_responses_file]
    if not all(os.path.exists(f) for f in required_files):
        print("错误: 一个或多个输入文件不存在。请检查提供的路径。")
        return
        
    print("="*80)
    print("🚀 开始学生能力分析...")
    
    evaluator = AbilityEvaluator(args.initial_tree_path)
    evaluator.evaluate_and_update(
        paths_filepath=args.paths_file,
        answers_filepath=args.answers_file,
        responses_filepath=args.student_responses_file
    )
    
    # 保存更新后的能力分数树
    evaluator.save_updated_tree(args.output_tree_path)
    
    # 计算并保存二级节点的统计数据
    evaluator.calculate_and_save_statistics(args.output_stats_path)
    
    print("\n✅ 学生能力分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()