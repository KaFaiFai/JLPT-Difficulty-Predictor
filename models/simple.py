"""
Simple model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel


class SimpleClassification(nn.Module):
    def __init__(self, num_class, vocab_size, embed_size=768, hidden_size=1000):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        return self.classifier(embedded)


def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    model = SimpleClassification(5, len(tokenizer)).to(device)

    line = "吾輩は猫である。"
    line2 = "国家公務員"
    inputs = tokenizer([line, line2], return_tensors="pt", padding="max_length").to(device)
    print(f"{len(tokenizer)=}")

    summary(model, **inputs)

    out = model(**inputs)
    print(model)


if __name__ == "__main__":
    test()
