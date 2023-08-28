"""
Simple model with positional encoding and 2 attentions
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel
import math


class PositionalEncoding(nn.Module):
    # adapted from pytorch https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, embed_size: int, dropout: float = 0, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # (B, seq_len, embed_size) -> (B, seq_len, embed_size)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SimpleAttention2(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_size=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = PositionalEncoding(embedding_size)
        self.attention1 = nn.MultiheadAttention(embedding_size, num_heads=1, batch_first=True)
        self.relu = nn.ReLU()
        self.attention2 = nn.MultiheadAttention(embedding_size, num_heads=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(embedding_size, num_class)

    def forward(self, input_ids, attention_mask, **kwargs):
        # (B, seq_len) -> (B, num_calss)

        # Embedding layer
        # (B, seq_len) -> (B, seq_len, embed_size)
        out = self.embedding(input_ids)
        out = self.positional_embedding(out)

        # Attention layer 1: self-attention
        # (B, seq_len, embed_size) -> (B, seq_len, embed_size)
        out, _ = self.attention1(out, out, out, key_padding_mask=attention_mask == 0)
        out = self.relu(out)

        # Attention layer 2: self-attention
        # (B, seq_len, embed_size) -> (B, seq_len, embed_size)
        out, _ = self.attention2(out, out, out, key_padding_mask=attention_mask == 0)
        out = self.relu(out)
        out = out[:, 0, ...]

        # Dropout layer
        # (B, seq_len, hidden_size * 2) -> (B, hidden_size * 2)
        out = self.dropout(out)

        # Fully connected layer
        # (B, hidden_size * 2) -> (B, num_calss)
        out = self.fc(out)
        return out

    def get_attention_output(self, input_ids, attention_mask, **kwargs):
        # (B, seq_len), (B, seq_len) -> (seq_len, B, hidden_size*2), (B, hidden_size*2, hidden_size*2)

        embedded = self.embedding(input_ids)
        embedded_pos = self.positional_embedding(embedded)
        mask = attention_mask == 0
        attention_output, attention_output_weights = self.attention1(
            embedded_pos, embedded_pos, embedded_pos, key_padding_mask=mask
        )
        return attention_output, attention_output_weights


def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    model = SimpleAttention2(5, len(tokenizer), 300).to(device)

    line = "吾輩は猫である。"
    line2 = "国家公務員"
    inputs = tokenizer([line, line2], return_tensors="pt", padding=True).to(device)
    print(f"{len(tokenizer)=}")

    summary(model, **inputs)

    out = model(**inputs)
    attention = model.get_attention_output(**inputs)
    print(model)


if __name__ == "__main__":
    test()
