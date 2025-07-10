import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np



class BertTextEmbedder(nn.Module):
    def __init__(self, pretrained_model, feature_dim=128):
        super(BertTextEmbedder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        self.feature_dim = feature_dim
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(768, feature_dim)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        pooled_output = self.pooling(last_hidden_state.transpose(1, 2)).squeeze(-1)  # [batch_size, 768]
        features = self.mlp(pooled_output)  # [batch_size, feature_dim]
        return features

    def process_texts(self, texts, max_samples=256):
        features_list = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            features = self(encoded_input['input_ids'], encoded_input['attention_mask'])
            features_list.append(features)
        all_features = torch.cat(features_list, dim=0)

        if all_features.shape[0] > max_samples:
            # Random sampling if there are more than 256 samples
            indices = np.random.choice(all_features.shape[0], max_samples, replace=False)
            selected_features = all_features[indices]
            # Alternatively, just select the first 256 samples
            # selected_features = all_features[:max_samples]
        else:
            selected_features = all_features

        return selected_features

