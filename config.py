# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class Config:
    # Diretórios do projeto
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Configurações do modelo
    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # Modelo BERT em português
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    
    # Labels para sentimentos
    SENTIMENT_LABELS = {
        0: "negativo",
        1: "neutro",
        2: "positivo"
    }
    
    def __init__(self):
        # Cria diretórios necessários
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)