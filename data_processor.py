# data_processor.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from config import Config

class TweetDataset(Dataset):
    """Dataset personalizado para tweets"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, 
                 max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokeniza o texto
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    """Classe para processamento de dados"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[DataLoader, 
                                                                           DataLoader]:
        """Prepara os dados para treinamento e teste"""
        # Carrega os dados
        df = pd.read_csv(data_path)
        
        # Divide em treino e teste
        texts = df['text'].values
        labels = df['sentiment'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Cria datasets
        train_dataset = TweetDataset(
            X_train, y_train, self.tokenizer, self.config.MAX_LENGTH
        )
        test_dataset = TweetDataset(
            X_test, y_test, self.tokenizer, self.config.MAX_LENGTH
        )
        
        # Cria dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE
        )
        
        return train_loader, test_loader
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Pré-processa um texto para predição"""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
