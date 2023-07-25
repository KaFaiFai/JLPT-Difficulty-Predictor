import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel


class SimpleAttention(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_class)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # LSTM layer
        lstm_output, _ = self.lstm(embedded)

        # TODO: add attention mask
        # Attention layer
        attention_output, _ = self.attention(
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1),
        )
        attention_output = attention_output.transpose(0, 1)

        # Dropout layer
        dropout_output = self.dropout(attention_output)

        # Fully connected layer
        fc_output = self.fc(dropout_output)
        output = torch.sigmoid(fc_output)

        return output


def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    net = SimpleAttention(5, len(tokenizer), 300, 500).to(device)

    line = "吾輩は猫である。"
    line2 = "国家公務員"
    inputs = tokenizer([line, line2], return_tensors="pt", padding="max_length").to(
        device
    )
    print(len(tokenizer))

    summary(net, input_data=inputs["input_ids"])

    # network_state = net.state_dict()
    # print("PyTorch model's state_dict:")
    # for layer, tensor in network_state.items():
    #     print(f"{layer:<45}: {tensor.size()}")


if __name__ == "__main__":
    test()
