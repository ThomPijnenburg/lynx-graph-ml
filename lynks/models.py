import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from lynks.layers import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.conv1 = GCNConv(in_channels=c_in, out_channels=c_out)
        self.conv2 = GCNConv(in_channels=c_out, out_channels=c_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LMEncoderClassificationModel(torch.nn.Module):
    """Text classification model using pretrained LM hidden state as embeddings for sentences.
    Follows implementation from Huggingface: https://github.com/huggingface/transformers/blob/bc6fe6fbcf6617612c50282fafb2ffb5d80cf880/src/transformers/models/distilbert/modeling_distilbert.py#L689
    HF implements CrossEntropyLoss for binary classification and BCEWithLogitsLoss for multi label.

    """

    def __init__(self, checkpoint: str, hidden_dim: int, n_labels: int, dropout: float = 0.3):
        super(LMEncoderClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(
            checkpoint)  # Language Model encoder
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(
            self.base_model.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask):
        lm_output = self.base_model(input_ids, attention_mask=attention_mask)
        hidden = lm_output[0]

        # pooled output
        x = hidden[:, 0, :]

        # You write your new head here
        x = self.dropout(x)
        x = self.projection(x)
        x = self.relu(x)
        x = self.classifier(x)  # logits
        return x
