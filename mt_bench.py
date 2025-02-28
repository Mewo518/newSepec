# full_evaluation.py
import torch
import time
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os

from torch.nn import functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行


class Benchmark:
    """基准方法：直接使用大模型生成（贪婪解码，遇到 eos_token 或达到最大长度终止）"""

    def __init__(self, model: str = "gpt2-xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text: str, max_length: int = 100) -> str:
        # 对输入进行编码
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        eos_token_id = self.tokenizer.eos_token_id

        generated_ids = input_ids.clone()
        # 循环生成后续 token，直至达到最大长度或生成 eos_token
        for _ in range(max_length - generated_ids.shape[1]):
            # 直接调用模型，获取 logits
            outputs = self.model(generated_ids)
            # 取最后一个 token 的 logits，并采用贪婪解码（取最大概率的 token）
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            # 拼接生成的 token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            # 如果生成了 eos_token，则提前结束
            if next_token_id.item() == eos_token_id:
                break
       # print(self.tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

class SpeculativeDecoder:
    """原始投机推理方法（基线），采用增量生成：
    - 小模型（draft_model）一次生成 gamma 个候选 token（贪婪解码）
    - 大模型验证每个 token 的正确性
    - 终止条件为达到最大长度或生成 eos_token
    """

    def __init__(self,
                 large_model: str = "gpt2-xl",
                 draft_model: str = "gpt2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model).to(device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(large_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.latency_stats = {"total": 0}

    def generate(self, input_text: str, max_length: int = 100, gamma: int = 5) -> str:
        """
        增量生成：
        - 草稿模型每次生成 gamma 个 token（贪婪解码）
        - 逐个 token 验证：对于草稿生成的每个 token，
          用大模型验证该 token 是否正确，若验证失败则采用大模型预测结果
        - 生成终止条件：达到 max_length 或生成 eos_token
        """
        # 编码输入
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        output_ids=input_ids
        eos_token_id = self.tokenizer.eos_token_id

        # 当生成序列未达到最大长度且最后一个 token 不是 eos_token 时，继续生成
        while output_ids.shape[1] < max_length:
            # 小模型一次生成 gamma 个候选 token，采用贪婪解码
            draft_outputs = self.draft_model.generate(
                input_ids,
                max_new_tokens=gamma,
                do_sample=False,  # 贪婪解码
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
            # softmax操作
            all_scores = torch.stack(draft_outputs.scores, dim=1)
            draft_logits = F.softmax(all_scores, dim=2)

            with torch.no_grad():  # 大模型
                target_probs = self.large_model(draft_outputs.sequences).logits
            target_logits=F.softmax(target_probs, dim=2)
            r = torch.rand(1, device=self.device)  # 生成随机数r，判断是否接受该token，是投机采样的关键参数
            #  接受/拒绝采样
            accepted = []

            for i in range(gamma):
                current_pos = input_ids.shape[1] + i
                draft_token = draft_outputs.sequences[0, current_pos] #当前判断的token id
                # draft_token_tensor = draft_token.unsqueeze(0)
                # 获取两个模型的概率估计

                q = draft_logits[0, i,draft_token]
                p = target_logits[0, current_pos - 1, draft_token]
                # 接受概率
                accept_prob = min(p / q, 1.0)
                if r <= accept_prob:
                    accepted.append(draft_token)
                    # accepted = torch.cat(accepted, draft_token_tensor)
                else:
                    # 拒绝时用目标模型重新采样
                    adjusted_probs = target_logits[0,  current_pos - 1] - draft_logits[0, i]
                    new_token=torch.argmax(adjusted_probs, dim=-1).unsqueeze(-1)
                    break
            if i==gamma-1:
                final_prob = target_logits[0, -1]
                new_token=torch.argmax(final_prob, dim=-1).unsqueeze(-1)
            accepted.append(new_token)
            #转为tensor
            accepted_tensor = torch.tensor(accepted).unsqueeze(0)
            accepted_tensor = accepted_tensor.to(input_ids.device)

            output_ids=torch.cat([output_ids, accepted_tensor],dim=-1)
        # print(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    """自适应投机解码（动态调整候选长度和模型）"""

    def __init__(self,
                 large_model="gpt2-xl",
                 draft_models=["gpt2", "gpt2-medium"],
                 gamma_ranges=[3, 5]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = 5

        # 加载大模型
        self.large_model = AutoModelForCausalLM.from_pretrained(
            large_model,
            # torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(large_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.draft_pool = {
            name: AutoModelForCausalLM.from_pretrained(name,
                                                       # torch_dtype=torch.float16,
                                                       device_map="auto")
            for name in draft_models
        }
        self.gamma_ranges = gamma_ranges
        self.complexity_cache = {}

    def _evaluate_complexity(self, text: str) -> float:
        """评估输入复杂度（基于长度和词汇多样性）"""
        if text in self.complexity_cache:
            return self.complexity_cache[text]

        # 计算特征
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        unique_ratio = len(set(tokens)) / (length + 1e-5)

        # 综合评分 (0-1)
        complexity = min(1.0, (length / 100) * 0.6 + (1 - unique_ratio) * 0.4)
        self.complexity_cache[text] = complexity
        return complexity

    def generate(self, prompt: str, max_length=200) -> str:
        complexity = self._evaluate_complexity(prompt)

        # 动态选择配置
        if complexity < 0.3:
            draft_model = self.draft_pool["gpt2"]
            gamma = self.gamma_ranges[1]  # 简单输入使用长候选
        elif complexity < 0.7:
            draft_model = self.draft_pool["gpt2-medium"]
            gamma = (self.gamma_ranges[0] + self.gamma_ranges[1]) // 2
        else:
            draft_model = self.draft_pool["gpt2-medium"]
            gamma = self.gamma_ranges[0]  # 复杂输入短候选

        self.draft_model = draft_model
        self.gamma = gamma

        return super().generate(prompt, max_length)


# ====================== 评估系统 ======================
class EvaluationSystem:
    """支持多数据集评估的评测系统"""

    def __init__(self):
        self.datasets = {
            "humaneval": self.load_humaneval,
            "gsm8k": self.load_gsm8k,
            "mt_bench": self.load_mt_bench
        }

    def load_humaneval(self):
        """加载编程题数据集"""
        dataset = load_dataset("openai_humaneval", split="test")
        return [{
            "prompt": ex["prompt"],
            "canonical_solution": ex["canonical_solution"],
            "test": ex["test"]
        } for ex in dataset]

    def load_gsm8k(self):
        """加载数学题数据集"""
        dataset = load_dataset("gsm8k", "main", split="test")
        return [{
            "question": ex["question"],
            "answer": ex["answer"].split("####")[1].strip()
        } for ex in dataset]

    def load_mt_bench(self):
        """加载对话数据集"""
        dataset = load_dataset("mt_bench", split="test")
        return [{"prompt": ex["prompt"]} for ex in dataset]

    def evaluate_code(self, problem: Dict, output: str) -> float:
        """代码题评估（简化版）"""
        try:
            # 提取模型生成的代码
            code = output.split("```python")[1].split("```")[0].strip()
            # 执行测试用例（注意：实际应使用沙箱环境）
            test_code = problem["test"].replace("candidate", code)
            exec(test_code)
            return 1.0
        except:
            return 0.0

    def evaluate_math(self, problem: Dict, output: str) -> float:
        """数学题评估"""
        try:
            pred = re.findall(r"\d+\.?\d*", output)[-1]
            return 1.0 if abs(float(pred) - float(problem["answer"])) < 1e-5 else 0.0
        except:
            return 0.0

    def run_evaluation(self, model, dataset_name: str, max_samples=10) -> Dict:
        """执行评估"""
        assert dataset_name in self.datasets, f"Unsupported dataset: {dataset_name}"
        dataset = self.datasets[dataset_name]()[:max_samples]

        results = []
        for example in tqdm(dataset, desc=f"Evaluating on {dataset_name}"):
            start_time = time.time()
            output = model.generate(example["prompt"])
            latency = time.time() - start_time

            # 计算准确率
            if dataset_name == "humaneval":
                score = self.evaluate_code(example, output)
            elif dataset_name == "gsm8k":
                score = self.evaluate_math(example, output)
            else:  # mt_bench
                score = len(output) / 500  # 简化的评分方法

            # 收集指标
            record = {
                "latency": latency,
                "score": score,
                "output": output
            }
            if hasattr(model, "metrics"):
                record.update({
                    "accept_rate": np.mean(model.metrics['accept_rates']),
                    "accepted_length": np.mean(model.metrics['accepted_lengths'])
                })
            results.append(record)

        return self._analyze_results(results, dataset_name)

    def _analyze_results(self, results: List[Dict], dataset_name: str) -> Dict:
        """分析评估结果"""
        return {
            "dataset": dataset_name,
            "accuracy": np.mean([r["score"] for r in results]),
            "avg_latency": np.mean([r["latency"] for r in results]),
            "accept_rate": np.mean([r.get("accept_rate", 0) for r in results]),
            "accepted_length": np.mean([r.get("accepted_length", 0) for r in results])
        }


# ====================== 可视化与执行 ======================
def plot_results(results: Dict):
    """可视化评估结果"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 精度对比
    datasets = list(results.keys())
    for idx, (model, data) in enumerate(results.items()):
        accs = [data[dset]["accuracy"] for dset in datasets]
        axs[0, 0].bar(np.arange(len(datasets)) + idx * 0.2, accs, width=0.2, label=model)
    axs[0, 0].set_title("Accuracy Comparison")
    axs[0, 0].set_xticks(np.arange(len(datasets)) + 0.3)
    axs[0, 0].set_xticklabels(datasets)

    # 延迟对比
    for idx, (model, data) in enumerate(results.items()):
        lats = [data[dset]["avg_latency"] for dset in datasets]
        axs[0, 1].bar(np.arange(len(datasets)) + idx * 0.2, lats, width=0.2, label=model)
    axs[0, 1].set_title("Latency Comparison")
    axs[0, 1].set_xticks(np.arange(len(datasets)) + 0.3)
    axs[0, 1].set_xticklabels(datasets)

    # 接受率分析
    models = list(results.keys())
    for idx, dset in enumerate(datasets):
        rates = [results[model][dset]["accept_rate"] for model in models]
        axs[1, 0].bar(np.arange(len(models)) + idx * 0.2, rates, width=0.2, label=dset)
    axs[1, 0].set_title("Acceptance Rate by Dataset")
    axs[1, 0].set_xticks(np.arange(len(models)) + 0.3)
    axs[1, 0].set_xticklabels(models)

    # 接收长度分布
    for model in models:
        lengths = [results[model][dset]["accepted_length"] for dset in datasets]
        axs[1, 1].scatter(datasets, lengths, label=model)
    axs[1, 1].set_title("Accepted Length Distribution")

    for ax in axs.flat:
        ax.legend()
    plt.tight_layout()
    plt.savefig("evaluation_results.png")


if __name__ == "__main__":
    evaluator = EvaluationSystem()
    models = {
        "Baseline": Benchmark(),
        "Standard Spec": SpeculativeDecoder(),
        "Adaptive Spec": AdaptiveSpeculativeDecoder()
    }

    all_results = {}
    for model_name, model in models.items():
        model_results = {}
        for dataset in ["humaneval"]:
        # for dataset in ["humaneval", "gsm8k", "mt_bench"]:
            print(f"\nEvaluating {model_name} on {dataset}...")
            result = evaluator.run_evaluation(model, dataset, max_samples=5)
            model_results[dataset] = result
        all_results[model_name] = model_results

    plot_results(all_results)
    print("\nEvaluation completed. Results saved to evaluation_results.png")