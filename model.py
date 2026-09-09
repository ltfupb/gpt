import torch
import torch.nn as nn
from tokenizer import load_tokenizer, encode

# embedding

d_model = 512
token_to_id, id_to_token, merges = load_tokenizer()

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

        x_normalized = x/rms*self.weight
        return x_normalized

norm = RMSNorm(d_model)
x = norm(x)

print(x.shape)