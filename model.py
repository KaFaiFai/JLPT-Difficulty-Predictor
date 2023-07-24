import torch
from  torch import nn
from transformers import AutoModel, AutoTokenizer, BertModel

class BERTClassification(nn.Module):
    def __init__(self, num_class=6):
        super().__init__()
        self.bert:BertModel = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        self.classifier = nn.Sequential(
nn.Dropout(0.3),nn.Linear(768, 768),nn.Dropout(0.3),nn.Linear(768, num_class),
        )
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output = self.classifier(output.pooler_output)
        return output

def test():
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BERTClassification(5).to(device)

    line = "吾輩は猫である。"
    line2 = "国家公務員"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    inputs = tokenizer([line,line2], return_tensors="pt", padding="max_length")
    summary(net, **inputs)


    # network_state = net.state_dict()
    # print("PyTorch model's state_dict:")
    # for layer, tensor in network_state.items():
    #     print(f"{layer:<45}: {tensor.size()}")


if __name__ == "__main__":
    test()