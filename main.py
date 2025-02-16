# main.py
from config import Config
from data_processor import DataProcessor
from model import SentimentClassifier
from trainer import ModelTrainer
import argparse
from pathlib import Path
import sys
import logging
import json

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sentiment_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description='Análise de Sentimento de Tweets com BERT'
    )
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='Modo de operação')
    parser.add_argument('--data', help='Caminho para arquivo de dados de treino')
    parser.add_argument('--text', help='Texto para análise')
    parser.add_argument('--model', help='Caminho para modelo salvo')
    
    args = parser.parse_args()
    
    try:
        # Inicializa configurações
        config = Config()
        
        # Inicializa processador de dados e modelo
        data_processor = DataProcessor(config)
        model = SentimentClassifier(config.MODEL_NAME)
        trainer = ModelTrainer(model, config)
        
        if args.mode == 'train':
            if not args.data:
                raise ValueError("Caminho dos dados de treino é necessário")
            
            logger.info("Iniciando treinamento...")
            
            # Prepara os dados
            train_loader, val_loader = data_processor.prepare_data(args.data)
            
            # Treina o modelo
            history = trainer.train(train_loader, val_loader)
            
            # Salva histórico de treinamento
            history_path = config.OUTPUT_DIR / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info("Treinamento concluído!")
            
        elif args.mode == 'predict':
            if not args.text:
                raise ValueError("Texto para análise é necessário")
            
            # Carrega modelo salvo se especificado
            if args.model:
                trainer.load_model(args.model)
            
            # Faz a predição
            result = trainer.predict(args.text, data_processor)
            
            print("\nResultado da Análise:")
            print(f"Texto: {result['text']}")
            print(f"Sentimento: {result['sentiment']}")
            print("\nProbabilidades:")
            for sentiment, prob in result['probabilities'].items():
                print(f"{sentiment}: {prob:.4f}")
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
