import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertModel


class BERTClassification(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.bert: BertModel = AutoModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(768, num_class),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.classifier(output.pooler_output)
        return output


def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BERTClassification(6).to(device)

    sentences = [
        "This is the first sentences",
        "I've been meaning to write for ages and finally today",
    ]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    summary(net, **inputs)

    out = net(**inputs)
    print(out.shape)

    print(net)
    # network_state = net.state_dict()
    # print("PyTorch model's state_dict:")
    # for layer, tensor in network_state.items():
    #     print(f"{layer:<45}: {tensor.size()}")


if __name__ == "__main__":
    test()
