# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    """Modelo de classificação de sentimento baseado em BERT"""
    
    def __init__(self, model_name: str, num_classes: int = 3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)