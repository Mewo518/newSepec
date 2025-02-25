import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SpeculativeDecoding:
    def __init__(self):
        self.large_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置pad token

        self.draft_models = {
            'tiny': self._build_draft_model(num_layers=2),
            'small': self._build_draft_model(num_layers=4),
            'medium': self._build_draft_model(num_layers=6)
        }

    def _build_draft_model(self, num_layers=4):
        return AutoModelForCausalLM.from_pretrained("gpt2", num_hidden_layers=num_layers)

    def evaluate_input_complexity(self, input_text):
        length = len(input_text.split())
        unique_words = len(set(input_text.split()))
        diversity = unique_words / length if length > 0 else 0
        return {'length': length, 'diversity': diversity}

    def select_draft_model(self, complexity):
        if complexity['length'] > 50 or complexity['diversity'] < 0.5:
            return self.draft_models['medium']
        elif complexity['length'] > 20:
            return self.draft_models[' small']
        else:
            return self.draft_models['tiny']

    def generate(self, input_text, max_length=50):
        complexity = self.evaluate_input_complexity(input_text)
        draft_model = self.select_draft_model(complexity)

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # 生成时获取logits
        generate_output = draft_model.generate(
            input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id  # 添加pad token设置
        )

        draft_tokens = generate_output.sequences
        scores = generate_output.scores  # 各生成步骤的logits

        # 计算每个token的置信度
        draft_confidences = []
        for step_score in scores:
            probs = torch.softmax(step_score, dim=-1)
            draft_confidences.append(probs.max().item())  # 取最大概率值

        # 自适应验证
        final_output = self.adaptive_verification(
            draft_tokens[0].tolist(),  # 转换为token列表
            draft_confidences
        )

        return self.tokenizer.decode(final_output)

    def adaptive_verification(self, draft_tokens, confidences, threshold=0.8):
        verified_tokens = []
        for i, (token, conf) in enumerate(zip(draft_tokens, confidences)):
            if conf >= threshold:
                verified_tokens.append(token)
            else:
                # 如果 verified_tokens 为空，直接使用当前 token
                if not verified_tokens:
                    verified_tokens.append(token)
                    continue

                # 使用大模型重新预测
                input_ids = torch.tensor([verified_tokens]).to(self.large_model.device)
                with torch.no_grad():
                    outputs = self.large_model(input_ids)
                next_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_logits).item()
                verified_tokens.append(next_token)

        # 处理剩余token（如果有）
        if len(verified_tokens) < len(draft_tokens):
            remaining = draft_tokens[len(verified_tokens):]
            verified_tokens.extend(remaining)

        return verified_tokens


if __name__ == "__main__":
    sd = SpeculativeDecoding()
    input_text = "The future of AI is"
    output = sd.generate(input_text)
    print("Generated Text:", output)