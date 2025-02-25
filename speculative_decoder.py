import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from tqdm import tqdm


class BaseSpeculativeDecoder:
    """原始投机推理方法（基线）"""

    def __init__(self,
                 large_model: str = "gpt2",
                 draft_model: str = "gpt2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model).to(device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(large_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.latency_stats = {"total": 0}

    def generate(self, input_text: str, max_length: int = 50) -> str:
        start_time = time.time()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # Draft 生成
        draft_outputs = self.draft_model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50
        )

        # 大模型验证
        with torch.no_grad():
            large_logits = self.large_model(draft_outputs).logits

        # 选择验证通过的序列
        verified = (large_logits.argmax(dim=-1) == draft_outputs).all(dim=1)
        output = draft_outputs[verified[0]] if verified.any() else draft_outputs[0]

        self.latency_stats["total"] = time.time() - start_time
        return self.tokenizer.decode(output, skip_special_tokens=True)


class AdvancedSpeculativeDecoder(BaseSpeculativeDecoder):
    """优化后的投机推理方法（支持动态模型选择和自适应验证）"""

    def __init__(self,
                 large_model: str = "gpt2",
                 draft_models: List[str] = ["gpt2", "gpt2-medium"],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        super().__init__(large_model, draft_models[0], device)
        self.draft_models = {
            name: AutoModelForCausalLM.from_pretrained(name).to(device)
            for name in draft_models
        }
        self.confidence_threshold = 0.85
        self.min_threshold = 0.6
        self.cache = {}

    def evaluate_input_complexity(self, input_text: str) -> Dict:
        if input_text in self.cache:
            return self.cache[input_text]

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.large_model(**inputs, output_hidden_states=True)

        complexity = {
            "length": inputs.input_ids.shape[1],
            "entropy": self._calculate_entropy(outputs.logits),
            "attention_variation": torch.std(outputs.hidden_states[-1]).item()
        }
        self.cache[input_text] = complexity
        return complexity

    def dynamic_model_selection(self, complexity: Dict) -> torch.nn.Module:
        if complexity["entropy"] > 2.0 or complexity["attention_variation"] > 0.5:
            return self.draft_models["gpt2-medium"]
        else:
            return self.draft_models["gpt2"]

    def generate(self, input_text: str, max_length: int = 50) -> str:
        start_time = time.time()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 动态选择 Draft 模型
        draft_model = self.dynamic_model_selection(self.evaluate_input_complexity(input_text))

        # Draft 生成
        draft_outputs = draft_model.generate(
            input_ids,
            max_length=max_length,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            do_sample=True,  # 启用采样
            num_return_sequences=3  # 生成多个候选
        )

        # 并行验证
        with torch.no_grad():
            large_logits = self.large_model(draft_outputs).logits

        # 选择最佳序列
        probs = torch.softmax(large_logits, dim=-1)
        avg_confidences = torch.max(probs, dim=-1).values.mean(dim=1)
        best_sequence = draft_outputs[torch.argmax(avg_confidences)]

        # 结果回退机制
        decoded_text = self.tokenizer.decode(best_sequence, skip_special_tokens=True)
        if not decoded_text.strip():
            outputs = self.large_model.generate(input_ids, max_length=max_length)
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.latency_stats["total"] = time.time() - start_time
        return decoded_text

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log2(probs + 1e-12)).mean().item()