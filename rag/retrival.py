import torch
import faiss
# Faiss 是一个用于高效相似性搜索和密集向量聚类的库。它包含在任意大小的向量组中搜索的算法
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载小模型（起草模型）和大模型（验证模型）
small_model_name = "facebook/opt-125m"
large_model_name = "facebook/opt-1.3b"

small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
small_model = AutoModelForCausalLM.from_pretrained(small_model_name)

large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)
large_model = AutoModelForCausalLM.from_pretrained(large_model_name)

# 初始化FAISS索引（存储知识检索的高质量Token Embedding）
d = 768  # 假设使用768维的embedding
index = faiss.IndexFlatL2(d)  # 使用L2距离的索引

# 假设我们有一些高质量的token embedding
high_quality_embeddings = torch.randn(1000, d)  # 这里应该是大模型生成的真实embedding
index.add(high_quality_embeddings.numpy())  # 添加到FAISS索引

# ---- 投机推理流程 ----
def speculative_decoding_with_retrieval(prompt, max_length=50, retrieve_every=5):
    input_ids = small_tokenizer(prompt, return_tensors="pt").input_ids
    generated_tokens = input_ids

    for step in range(max_length):
        outputs = small_model(generated_tokens, output_hidden_states=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=-1)

        if step % retrieve_every == 0 and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1][:, -1, :].detach().numpy()
            _, retrieved_idx = index.search(last_hidden_state, k=1)

            # 修正 numpy.int64 转换问题
            retrieved_token_id = torch.tensor([[retrieved_idx[0][0].item()]], dtype=torch.long)
            generated_tokens = torch.cat((generated_tokens, retrieved_token_id), dim=-1)

        verification_outputs = large_model(generated_tokens)
        verified_token = torch.argmax(verification_outputs.logits[:, -1, :], dim=-1, keepdim=True)

        if not torch.equal(verified_token, next_token):
            generated_tokens[:, -1] = verified_token.squeeze()

    return small_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


# 测试
prompt = "Once upon a time"
output_text = speculative_decoding_with_retrieval(prompt)
print(output_text)
