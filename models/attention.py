"""
Simple model with LSTM and attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel


class SimpleAttention(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_size=300, hidden_size=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_class)

    def forward(self, input_ids, attention_mask, **kwargs):
        # (B, seq_len) -> (B, num_calss)

        # Embedding layer
        # (B, seq_len) -> (B, seq_len, embed_size)
        out = self.embedding(input_ids)

        # LSTM layer
        # (B, seq_len, embed_size) -> (B, seq_len, hidden_size * 2) for bidirectional
        out, _ = self.lstm(out)
        out = self.relu(out)

        # Attention layer: self-attention
        # (B, seq_len, hidden_size * 2) -> (B, seq_len, hidden_size * 2)
        out, _ = self.attention(out, out, out, key_padding_mask=attention_mask == 0)
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
        lstm_output, _ = self.lstm(embedded)
        relu_output = self.relu(lstm_output)
        mask = attention_mask == 0
        attention_output, attention_output_weights = self.attention(
            lstm_output, lstm_output, lstm_output, key_padding_mask=mask
        )
        return attention_output, attention_output_weights


def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    model = SimpleAttention(5, len(tokenizer), 300, 500).to(device)

    line = "吾輩は猫である。"
    line2 = "国家公務員"
    inputs = tokenizer([line, line2], return_tensors="pt", padding="max_length").to(device)
    print(f"{len(tokenizer)=}")

    summary(model, **inputs)

    out = model(**inputs)
    print(model)


if __name__ == "__main__":
    test()
