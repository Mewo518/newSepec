import json
import numpy as np

def load_test_dataset(path: str) -> List[str]:
    """加载测试数据集（示例格式：每行一个文本）"""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def calculate_bleu(reference: str, generated: str) -> float:
    """计算 BLEU-4 分数"""
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([reference.split()], generated.split(), weights=(0.25, 0.25, 0.25, 0.25))

def calculate_rouge(reference: str, generated: str) -> float:
    """计算 ROUGE-L F1 分数"""
    from rouge import Rouge
    return Rouge().get_scores(generated, reference)[0]["rouge-l"]["f"]