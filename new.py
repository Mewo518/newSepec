import torch
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from speculative_decoder import SpeculativeDecoder


class Benchmark:
    """基准方法：直接使用大模型生成"""

    def __init__(self, model: str = "gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====================== 自适应投机推理框架 ======================
class AdaptiveSpeculativeDecoder:
    """基于输入复杂度动态调整的自适应投机推理框架"""

    def __init__(self,
                 large_model: str = "gpt2-medium",  # 大模型（验证模型）
                 draft_pool: List[str] = ["gpt2", "gpt2-medium"],  # Draft模型池
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device

        # 加载大模型
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model).to(device)

        # 加载Draft模型池
        self.draft_models = {
            name: AutoModelForCausalLM.from_pretrained(name).to(device)
            for name in draft_pool
        }

        # Tokenizer设置
        self.tokenizer = AutoTokenizer.from_pretrained(large_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 自适应参数
        self.complexity_thresholds = {
            'low': {'model': 'gpt2', 'gamma': 5},
            'medium': {'model': 'gpt2-medium', 'gamma': 3},
            'high': {'model': 'gpt2-medium', 'gamma': 1}
        }

    def _evaluate_complexity(self, input_ids: torch.Tensor) -> str:
        """评估输入复杂度（核心创新点）"""
        # 特征1：输入长度
        seq_len = input_ids.shape[-1]

        # 特征2：词汇多样性（使用大模型隐层状态）
        with torch.no_grad():
            outputs = self.large_model(input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            diversity = torch.std(last_hidden.mean(dim=1))

        # 动态分类逻辑
        if seq_len > 100 or diversity > 0.5:
            return 'high'
        elif seq_len > 50 or diversity > 0.3:
            return 'medium'
        else:
            return 'low'

    def _adaptive_draft(self, input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """自适应起草阶段"""
        # 评估复杂度
        complexity = self._evaluate_complexity(input_ids)
        config = self.complexity_thresholds[complexity]

        # 选择Draft模型
        draft_model = self.draft_models[config['model']]
        gamma = config['gamma']

        # 生成候选
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids.clone()

        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(current_ids).logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token)
            draft_probs.append(probs.gather(-1, next_token))
            current_ids = torch.cat([current_ids, next_token], dim=-1)

        return draft_tokens, draft_probs, config

    def _adaptive_verify(self,
                         input_ids: torch.Tensor,
                         draft_tokens: List[torch.Tensor],
                         draft_probs: List[torch.Tensor],
                         config: Dict) -> torch.Tensor:
        """自适应验证阶段"""
        # 动态调整置信阈值
        base_threshold = 0.7
        if config['gamma'] > 3:
            threshold = base_threshold * 0.9  # 长序列降低阈值
        else:
            threshold = base_threshold * 1.1  # 短序列提高阈值

        # 并行验证
        candidate_ids = torch.cat([input_ids] + draft_tokens, dim=-1)
        with torch.no_grad():
            large_logits = self.large_model(candidate_ids).logits[0, input_ids.shape[-1]:, :]
        # 接受概率计算（动态调整）
        accept_probs = []
        for i in range(len(draft_tokens)):
            q = torch.softmax(large_logits[i], dim=-1)
            p = draft_probs[i]
            ratio = (q / (p + 1e-10)).gather(-1, draft_tokens[i])
            accept_prob = torch.min(torch.tensor(1.0), ratio * threshold)  # 动态阈值
            accept_probs.append(accept_prob)

        # 确定接受位置
        accept_mask = torch.rand(len(accept_probs)).to(self.device) < torch.cat(accept_probs)
        n = accept_mask.sum().item()

        # 自适应回退
        if n < len(draft_tokens):
            residual_probs = torch.softmax(large_logits[n] - draft_probs[n], dim=-1)
            corrected_token = torch.multinomial(residual_probs, 1)
            return candidate_ids[:, :input_ids.shape[-1] + n], corrected_token
        else:
            return candidate_ids, None

    def generate(self, input_text: str, max_length: int = 100) -> str:
        """完整生成流程"""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        for _ in range(max_length - input_ids.shape[1]):
            # 自适应起草
            draft_tokens, draft_probs, config = self._adaptive_draft(input_ids)

            # 自适应验证
            new_ids, corrected = self._adaptive_verify(input_ids, draft_tokens, draft_probs, config)

            # 更新输入
            input_ids = torch.cat([new_ids, corrected], dim=-1) if corrected is not None else new_ids

            if input_ids[0, -1] == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ====================== 实验对比框架 ======================
class ExperimentSystem:
    """实验系统：对比三种方法"""

    def __init__(self, dataset: List[str]):
        self.dataset = dataset
        self.results = []

    def _evaluate_quality(self, ref: str, hyp: str) -> Dict:
        """质量评估（含异常处理）"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge import Rouge

            # BLEU with smoothing
            bleu = sentence_bleu([ref.split()], hyp.split(),
                                 weights=(0.25, 0.25, 0.25, 0.25),
                                 smoothing_function=SmoothingFunction().method1)

            # ROUGE-L
            rouge = Rouge().get_scores(hyp, ref)[0]["rouge-l"]["f"]
        except:
            bleu, rouge = 0.0, 0.0

        return {"bleu": bleu, "rouge": rouge}

    def run_experiments(self, num_samples=20):
        """运行对比实验"""
        methods = {
            "Baseline (Large Model)": Benchmark(model="gpt2-medium"),
            "Standard Speculative": SpeculativeDecoder(),
            "Our Adaptive Method": AdaptiveSpeculativeDecoder()
        }

        for text in tqdm(self.dataset[:num_samples]):
            record = {"text": text}
            for name, model in methods.items():
                start = time.time()
                output = model.generate(text)
                latency = time.time() - start
                metrics = self._evaluate_quality(text, output)
                record[name] = {
                    "output": output,
                    "latency": latency,
                    **metrics
                }
            self.results.append(record)

    def analyze_results(self):
        """结果分析与可视化"""
        metrics = ["bleu", "rouge", "latency"]
        method_names = ["Baseline (Large Model)", "Standard Speculative", "Our Adaptive Method"]

        # 统计平均指标
        avg_metrics = {name: {m: [] for m in metrics} for name in method_names}
        for record in self.results:
            for name in method_names:
                for m in metrics:
                    avg_metrics[name][m].append(record[name][m])

        # 打印结果
        print(f"{'Method':<25} | {'BLEU':<6} | {'ROUGE':<6} | {'Latency(s)':<8}")
        print("-" * 50)
        for name in method_names:
            bleu = np.mean(avg_metrics[name]['bleu'])
            rouge = np.mean(avg_metrics[name]['rouge'])
            latency = np.mean(avg_metrics[name]['latency'])
            print(f"{name:<25} | {bleu:.4f} | {rouge:.4f} | {latency:.2f}    ")

        # 可视化
        plt.figure(figsize=(12, 5))

        # BLEU-Latency散点图
        plt.subplot(121)
        for name in method_names:
            bleus = [r[name]['bleu'] for r in self.results]
            lats = [r[name]['latency'] for r in self.results]
            plt.scatter(lats, bleus, label=name, alpha=0.6)
        plt.xlabel('Latency (s)')
        plt.ylabel('BLEU Score')
        plt.legend()

        # 速度提升直方图
        plt.subplot(122)
        baseline_lats = [r["Baseline (Large Model)"]['latency'] for r in self.results]
        adaptive_lats = [r["Our Adaptive Method"]['latency'] for r in self.results]
        speedup_ratios = [b / a for b, a in zip(baseline_lats, adaptive_lats)]
        plt.hist(speedup_ratios, bins=15, alpha=0.7)
        plt.xlabel('Speedup Ratio (Baseline/Adaptive)')
        plt.ylabel('Frequency')
        plt.title(f'Average Speedup: {np.mean(speedup_ratios):.2f}x')

        plt.tight_layout()
        plt.savefig('comparison_results.png')


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 准备测试数据
    test_data = [
        "Recent advances in AI demonstrate that",  # 中等复杂度
        "The",  # 低复杂度
        "In the context of modern machine learning, transformer-based models have revolutionized"  # 高复杂度
    ]

    # 运行实验
    exp_system = ExperimentSystem(test_data)
    exp_system.run_experiments()
    exp_system.analyze_results()