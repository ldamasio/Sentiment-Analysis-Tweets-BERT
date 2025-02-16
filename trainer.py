# trainer.py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Tuple
import numpy as np
import logging
from config import Config
from data_processor import DataProcessor
from torch.utils.data import DataLoader

class ModelTrainer:
    """Classe para treinamento e avaliação do modelo"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Treina o modelo"""
        best_accuracy = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0
            
            # Loop de treinamento
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Avaliação
            val_loss, accuracy = self.evaluate(val_loader)
            
            # Atualiza histórico
            avg_train_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)
            
            self.logger.info(
                f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, '
                f'Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}'
            )
            
            # Salva o melhor modelo
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model('best_model.pt')
        
        return history
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Avalia o modelo"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, accuracy
    
    def predict(self, text: str, data_processor: DataProcessor) -> Dict:
        """Faz predição para um texto"""
        self.model.eval()
        
        # Pré-processa o texto
        inputs = data_processor.preprocess_text(text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Faz a predição
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Prepara o resultado
        result = {
            'text': text,
            'sentiment': self.config.SENTIMENT_LABELS[prediction.item()],
            'probabilities': {
                label: prob.item()
                for label, prob in zip(
                    self.config.SENTIMENT_LABELS.values(),
                    probabilities[0]
                )
            }
        }
        
        return result
    
    def save_model(self, filename: str):
        """Salva o modelo"""
        path = self.config.MODELS_DIR / filename
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, filename: str):
        """Carrega o modelo"""
        path = self.config.MODELS_DIR / filename
        self.model.load_state_dict(torch.load(path))