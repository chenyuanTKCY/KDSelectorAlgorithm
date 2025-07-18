import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# text_mlp = TextMLP(768)


def process_text(texts, device, output_dim, bert_model, mlp_text, LLM_mode):
    # print("Original texts:", texts)

    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}

    if LLM_mode == 'eval':
        with torch.no_grad():
            bert_model.to(device)
            outputs = bert_model(**encoded_inputs)
    else:
        bert_model.to(device)
        outputs = bert_model(**encoded_inputs)

    mlp_text.to(device)
    text_features = outputs.last_hidden_state[:, 0, :]
    text_features = mlp_text(text_features)


    return text_features


