import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from speculative_decoder import BaseSpeculativeDecoder, AdvancedSpeculativeDecoder


class ExperimentRunner:
    def __init__(self, dataset_path: str = "data/test_samples.txt"):
        self.dataset = self.load_dataset(dataset_path)
        self.metrics = {"baseline": [], "optimized": []}

    def load_dataset(self, path: str) -> List[str]:
        """加载数据集并过滤空行"""
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def safe_bleu(self, reference: str, generated: str) -> float:
        """带异常处理的BLEU计算"""
        from nltk.translate.bleu_score import sentence_bleu  #nltk：自然语言处理工具包，提供文本处理、分词、句子拆分等功能
        try:
            return sentence_bleu([reference.split()], generated.split(),
                                 weights=(0.25, 0.25, 0.25, 0.25))
        except:
            return 0.0

    def safe_rouge(self, reference: str, generated: str) -> float:
        """带异常处理的ROUGE计算"""
        from rouge import Rouge  #rouge：用于计算 ROUGE 指标的工具包，常用于评估生成文本的质量
        try:
            return Rouge().get_scores(generated, reference)[0]["rouge-l"]["f"]
        except:
            return 0.0

    def run_experiment(self, num_samples: int = 10):
        """运行实验并收集指标"""
        Path("results").mkdir(exist_ok=True)

        for text in tqdm(self.dataset[:num_samples], desc="Processing"):
            # 基线方法
            baseline = BaseSpeculativeDecoder()
            baseline_output = baseline.generate(text, max_length=50)
            self.metrics["baseline"].append({
                "bleu": self.safe_bleu(text, baseline_output),
                "rouge": self.safe_rouge(text, baseline_output),
                "latency": baseline.latency_stats["total"]
            })

            # 优化方法
            optimized = AdvancedSpeculativeDecoder()
            optimized_output = optimized.generate(text, max_length=50)
            self.metrics["optimized"].append({
                "bleu": self.safe_bleu(text, optimized_output),
                "rouge": self.safe_rouge(text, optimized_output),
                "latency": optimized.latency_stats["total"]
            })

        # 结果分析
        self.analyze_results()

    def analyze_results(self):
        """结果分析与可视化"""
        # 计算平均值
        baseline_bleu = np.mean([m["bleu"] for m in self.metrics["baseline"]])
        optimized_bleu = np.mean([m["bleu"] for m in self.metrics["optimized"]])
        baseline_latency = np.mean([m["latency"] for m in self.metrics["baseline"]])
        optimized_latency = np.mean([m["latency"] for m in self.metrics["optimized"]])

        # 打印结果
        print(f"\n{'Metric':<15} | {'Baseline':<10} | {'Optimized':<10}")
        print("-" * 40)
        print(f"{'BLEU':<15} | {baseline_bleu:.4f}     | {optimized_bleu:.4f}")
        print(f"{'ROUGE-L':<15} | {np.mean([m['rouge'] for m in self.metrics['baseline']]):.4f}     | "
              f"{np.mean([m['rouge'] for m in self.metrics['optimized']]):.4f}")
        print(f"{'Latency (s)':<15} | {baseline_latency:.2f}     | {optimized_latency:.2f}")
        print(f"\nSpeedup Ratio: {baseline_latency / optimized_latency:.2f}x")

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [m["latency"] for m in self.metrics["baseline"]],
            [m["bleu"] for m in self.metrics["baseline"]],
            label="Baseline", alpha=0.6
        )
        plt.scatter(
            [m["latency"] for m in self.metrics["optimized"]],
            [m["bleu"] for m in self.metrics["optimized"]],
            label="Optimized", alpha=0.6
        )
        plt.xlabel("Latency (seconds)")
        plt.ylabel("BLEU Score")
        plt.title("Quality vs. Latency Tradeoff")
        plt.legend()
        plt.savefig("results/quality_vs_latency.png")
        plt.close()


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiment(num_samples=5)  # 使用前5个样本快速测试