import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Lógica de Dados ---

class CovidDataset(Dataset):
    """Dataset customizado para sequências de DNA do COVID-19."""
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding=False,
            max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

def load_and_prepare_data(config, tokenizer):
    """Carrega e prepara os datasets com base na configuração."""
    print("Carregando datasets...")
    data_config = config['data']
    
    train_df = pd.read_csv(data_config['train_path'])
    dev_df = pd.read_csv(data_config['dev_path'])
    test_df = pd.read_csv(data_config['test_path'])

    if data_config.get('train_sample_size'):
        train_df = train_df.sample(n=data_config['train_sample_size'], random_state=42)
    if data_config.get('dev_sample_size'):
        dev_df = dev_df.sample(n=data_config['dev_sample_size'], random_state=42)
    if data_config.get('test_sample_size'):
        test_df = test_df.sample(n=data_config['test_sample_size'], random_state=42)
        
    print(f"Dados carregados: Treino({len(train_df):,}), Dev({len(dev_df):,}), Teste({len(test_df):,})")

    max_length = config['model']['max_length']
    train_dataset = CovidDataset(train_df['sequence'].values, train_df['label'].values, tokenizer, max_length)
    dev_dataset = CovidDataset(dev_df['sequence'].values, dev_df['label'].values, tokenizer, max_length)
    test_dataset = CovidDataset(test_df['sequence'].values, test_df['label'].values, tokenizer, max_length)

    return train_dataset, dev_dataset, test_dataset

# --- Lógica de Métricas ---

def compute_metrics(eval_pred):
    """Calcula a acurácia para o Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0] if isinstance(predictions, tuple) else predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def generate_final_reports(trainer, test_dataset, output_dir):
    """Gera e salva relatórios de classificação e matriz de confusão."""
    print("\nGerando relatórios finais...")
    predictions, labels, _ = trainer.predict(test_dataset)
    final_preds = np.argmax(predictions[0] if isinstance(predictions, tuple) else predictions, axis=1)

    report = classification_report(labels, final_preds, target_names=[f"Classe {i}" for i in range(9)])
    cm = confusion_matrix(labels, final_preds)
    
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)
    np.savetxt(f"{output_dir}/confusion_matrix.txt", cm, fmt="%d")
    print(f"\nRelatórios salvos em: {output_dir}") 