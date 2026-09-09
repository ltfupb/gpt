import torch
import torch.nn as nn
from tokenizer import load_tokenizer, encode

d_model = 512 # 512차원
num_heads = 8 # 헤드 수
token_to_id, id_to_token, merges = load_tokenizer()

# embedding

embedding = nn.Embedding(len(token_to_id), d_model)

text = "아이오니아가 부른다"
ids = encode(text)

input_ids = torch.tensor(ids)

x = embedding(input_ids)

# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        eps = 1e-6

        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + eps
        )

        x_normalized = x / rms * self.weight
        return x_normalized

norm = RMSNorm(d_model)
x = norm(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(q.shape[0], self.num_heads, self.d_head)
        k = k.reshape(k.shape[0], self.num_heads, self.d_head)
        v = v.reshape(v.shape[0], self.num_heads, self.d_head)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.d_head ** 0.5)

        seq_len = q.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)

        out = attention @ v

        out = out.transpose(0, 1)
        out = out.reshape(out.shape[0], -1)
        out = self.out_proj(out)

        return out