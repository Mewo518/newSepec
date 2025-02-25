import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class AdvancedSpeculativeDecoder:
    def __init__(self,
                 large_model_name: str = "gpt2",  # 改用更小的模型便于测试
                 draft_model_names: List[str] = ["gpt2", "gpt2-medium"],  # 实际需预训练的 Draft 模型
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device

        # 加载大模型并显式设置 pad_token
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(large_model_name)
        if self.tokenizer.pad_token is None:  # 关键修复：确保存在 pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 Draft 模型池
        self.draft_models = {
            name: AutoModelForCausalLM.from_pretrained(name).to(device)
            for name in draft_model_names
        }

        # 初始化其他参数
        self.confidence_threshold = 0.85
        self.min_threshold = 0.6
        self.latency_stats = {"total": 0, "draft": 0, "verify": 0}
        self.cache = {}

    # -------------------- 核心逻辑 --------------------
    def evaluate_input_complexity(self, input_text: str) -> Dict:
        """输入复杂度评估（简化版）"""
        if input_text in self.cache:
            return self.cache[input_text]

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.large_model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        complexity = {
            "length": inputs.input_ids.shape[1],
            "entropy": self._calculate_entropy(outputs.logits),
            "attention_variation": torch.std(last_hidden).item()
        }
        self.cache[input_text] = complexity
        return complexity

    def dynamic_model_selection(self, complexity: Dict) -> torch.nn.Module:
        """动态选择 Draft 模型（规则引擎）"""
        if complexity["entropy"] > 2.0 or complexity["attention_variation"] > 0.5:
            return self.draft_models["gpt2-medium"]
        else:
            return self.draft_models["gpt2"]

    def speculative_generate(self, input_text: str, max_length: int = 50) -> str:
        start_time = time.time()
        self.latency_stats = {"total": 0, "draft": 0, "verify": 0}

        # 输入编码和格式校验
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        if input_ids.dim() == 1:  # 确保输入是二维张量 [1, seq_len]
            input_ids = input_ids.unsqueeze(0)

        # 生成 attention_mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        pad_token_id = self.tokenizer.eos_token_id

        # Draft 生成
        draft_time = time.time()
        draft_model = self.dynamic_model_selection(self.evaluate_input_complexity(input_text))
        draft_outputs = draft_model.generate(
            input_ids,
            max_length=max_length,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id
        )
        self.latency_stats["draft"] = time.time() - draft_time

        # 并行验证
        verify_time = time.time()
        best_sequence = self._parallel_verify(draft_outputs, input_ids)
        self.latency_stats["verify"] = time.time() - verify_time

        # 动态阈值调整（防止除零错误）
        if self.latency_stats["total"] > 1e-6:
            self._adaptive_threshold_adjustment()

        self.latency_stats["total"] = time.time() - start_time
        return self.tokenizer.decode(best_sequence, skip_special_tokens=True)

    def _parallel_verify(self, draft_sequences: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """批量验证候选序列"""
        with torch.no_grad():
            large_logits = self.large_model(draft_sequences).logits
        probs = torch.softmax(large_logits, dim=-1)
        avg_confidences = torch.max(probs, dim=-1).values.mean(dim=1)
        return draft_sequences[torch.argmax(avg_confidences)]

    def _adaptive_threshold_adjustment(self):
        verify_time_ratio = self.latency_stats["verify"] / self.latency_stats["total"]
        if verify_time_ratio > 0.7:
            self.confidence_threshold = max(self.confidence_threshold * 0.95, self.min_threshold)
        else:
            self.confidence_threshold = min(self.confidence_threshold * 1.05, 0.95)

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        """计算熵"""
        probs = torch.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log2(probs + 1e-12)).mean().item()

    def reset_stats(self):
        self.latency_stats = {"total": 0, "draft": 0, "verify": 0}


# -------------------- 测试用例 --------------------
if __name__ == "__main__":
    # 初始化解码器（使用小模型加速测试）
    decoder = AdvancedSpeculativeDecoder(
        large_model_name="gpt2",
        draft_model_names=["gpt2", "gpt2-medium"],
        device="cpu"  # 使用 CPU 运行便于测试
    )

    text = "Artificial intelligence is"
    print("Input:", text)

    # 首次生成（包含模型加载时间）
    output = decoder.speculative_generate(text, max_length=30)
    print("\nGenerated Text:", output)
    print(f"Latency: {decoder.latency_stats['total']:.2f}s")