import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from tqdm import tqdm


class BaseSpeculativeDecoder:
    """原始投机推理方法（基线）"""

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

    def generate(self, input_text: str, max_length: int = 50) -> str:
        start_time = time.time()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # Draft 生成
        draft_outputs = self.draft_model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
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


class SpeculativeDecoder:
    """标准投机解码实现（小模型起草 + 大模型验证）"""

    def __init__(self,
                 large_model: str = "gpt2-medium",
                 draft_model: str = "gpt2",
                 gamma: int = 5,  # 最大候选长度
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.gamma = gamma

        # 大模型（验证模型）
        self.large_model = AutoModelForCausalLM.from_pretrained(large_model).to(device)

        # 小模型（起草模型）
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model).to(device)

        # Tokenizer设置
        self.tokenizer = AutoTokenizer.from_pretrained(large_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.latency_stats = {"total": 0}
    def _draft_step(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """起草阶段：小模型生成gamma个候选token"""
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids.clone()

        for _ in range(self.gamma):
            # 获取小模型的概率分布
            with torch.no_grad():
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_token)
            draft_probs.append(probs.gather(-1, next_token))
            current_ids = torch.cat([current_ids, next_token], dim=-1)

        return draft_tokens, draft_probs

    def _verify_step(self, input_ids: torch.Tensor, draft_tokens: List[torch.Tensor],draft_probs) -> torch.Tensor:
        """验证阶段：大模型并行验证候选"""
        # 构建候选序列
        candidate_ids = torch.cat([input_ids] + draft_tokens, dim=-1)

        # 获取大模型概率
        with torch.no_grad():
            outputs = self.large_model(candidate_ids)
            large_logits = outputs.logits[0, input_ids.shape[-1]:, :] # 只取候选位置的logits

        # 计算接受概率
        accept_probs = []
        for i in range(len(draft_tokens)):
            q = torch.softmax(large_logits[i], dim=-1)
            p = draft_probs[i]
            ratio = (q / p).gather(-1, draft_tokens[i])
            accept_prob = torch.min(torch.tensor(1.0).to(self.device), ratio)
            accept_probs.append(accept_prob)

        # 确定接受位置
        accept_mask = torch.rand(len(accept_probs)).to(self.device) < torch.cat(accept_probs)
        accept_index = torch.where(accept_mask)[0]
        n = accept_index[-1] + 1 if len(accept_index) > 0 else 0

        # 处理回退情况
        if n < len(draft_tokens):
            residual_probs = torch.softmax(large_logits[n] - draft_probs[n], dim=-1)
            corrected_token = torch.multinomial(residual_probs, 1)
            return candidate_ids[:, :input_ids.shape[-1] + n], corrected_token
        else:
            return candidate_ids, None

    def generate(self, input_text: str, max_length: int = 50) -> str:
        """完整生成流程"""
        start_time = time.time()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        for _ in range(max_length - input_ids.shape[1]):
            # 起草阶段
            draft_tokens, draft_probs = self._draft_step(input_ids)

            # 验证阶段
            new_ids, corrected = self._verify_step(input_ids, draft_tokens, draft_probs)

            # 更新输入
            if corrected is not None:
                input_ids = torch.cat([new_ids, corrected], dim=-1)
            else:
                input_ids = new_ids

            if input_ids[0, -1] == self.tokenizer.eos_token_id:
                break
        self.latency_stats["total"] = time.time() - start_time
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


class AdvancedSpeculativeDecoder(BaseSpeculativeDecoder):
    """优化后的投机推理方法（支持动态模型选择和自适应验证）"""

    def __init__(self,
                 large_model: str = "gpt2-xl",
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