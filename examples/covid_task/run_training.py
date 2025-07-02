#!/usr/bin/env python3
"""
Script unificado para treinamento e fine-tuning do DNABERT-2 na tarefa COVID.
Utiliza perfis de um arquivo config.yaml para configurar a execução.

Uso:
  cd examples/covid_task
  python run_training.py --profile a100_finetune
"""

import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
import json
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
from datetime import datetime

# --- Classes e Funções de Suporte ---

class CovidDataset(Dataset):
    """Dataset customizado para sequências COVID-19."""
    def __init__(self, sequences, labels, tokenizer, max_length):
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
            sequence, truncation=True, padding=False, max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

def compute_metrics(eval_pred):
    """Calcula métricas de acurácia."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# --- Pipeline Principal ---

def main(profile_name):
    # Carregar configuração
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    general_config = config['general']
    data_paths = config['data_paths']
    profile = config['profiles'][profile_name]
    
    print(f"🧬 Iniciando Treinamento com Perfil: {profile_name}")
    print(f"   Descrição: {profile['description']}")
    print("-" * 60)

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Carregar dados
    print("\n📁 Carregando dataset...")
    train_df = pd.read_csv(os.path.join("../..", data_paths['train']))
    dev_df = pd.read_csv(os.path.join("../..", data_paths['dev']))
    test_df = pd.read_csv(os.path.join("../..", data_paths['test']))
    
    if not profile.get('use_full_dataset', True):
        sample_size = profile.get('sample_size', 500)
        print(f"🔪 Usando uma amostra de {sample_size} para cada dataset...")
        train_df = train_df.sample(n=sample_size, random_state=42)
        dev_df = dev_df.sample(n=sample_size, random_state=42)
        test_df = test_df.sample(n=sample_size, random_state=42)

    print(f"✅ Dados carregados: Treino={len(train_df)}, Dev={len(dev_df)}, Teste={len(test_df)}")

    # Carregar modelo e tokenizador
    print("\n🤖 Carregando DNABERT-2...")
    tokenizer = AutoTokenizer.from_pretrained(general_config['model_name'], trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        general_config['model_name'],
        trust_remote_code=True,
        num_labels=general_config['num_labels']
    )
    model.to(device)
    print(f"✅ Modelo {general_config['model_name']} carregado.")

    # Criar datasets
    print("\n🔄 Preparando datasets...")
    max_length = profile['max_seq_length']
    train_dataset = CovidDataset(train_df['sequence'].values, train_df['label'].values, tokenizer, max_length)
    dev_dataset = CovidDataset(dev_df['sequence'].values, dev_df['label'].values, tokenizer, max_length)
    test_dataset = CovidDataset(test_df['sequence'].values, test_df['label'].values, tokenizer, max_length)
    print(f"✅ Datasets criados com max_length={max_length}")

    # Configurar argumentos de treinamento
    output_dir = os.path.join(
        general_config['base_output_dir'], 
        f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Adicionar parâmetros padrão que não mudam entre perfis
    training_args_dict = profile['training_args']
    
    # Garantir que a learning_rate seja um float
    if 'learning_rate' in training_args_dict:
        training_args_dict['learning_rate'] = float(training_args_dict['learning_rate'])

    training_args_dict.setdefault('weight_decay', 0.01)
    training_args_dict.setdefault('logging_steps', 200)
    training_args_dict.setdefault('eval_steps', 1000)
    training_args_dict.setdefault('save_steps', 1000)
    training_args_dict.setdefault('save_strategy', "steps")
    training_args_dict.setdefault('evaluation_strategy', "steps")
    training_args_dict.setdefault('save_total_limit', 2)
    training_args_dict.setdefault('warmup_steps', 500)
    
    # Lidar com callbacks separadamente, pois não são argumentos de TrainingArguments
    callbacks_config = training_args_dict.pop('callbacks', None)
    callbacks = []
    if callbacks_config and 'early_stopping_patience' in callbacks_config:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=callbacks_config['early_stopping_patience']
        ))
        # Adicionar 'load_best_model_at_end' é uma boa prática com early stopping
        training_args_dict['load_best_model_at_end'] = True

    training_args = TrainingArguments(output_dir=output_dir, **training_args_dict)
    
    # Configurar Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Treinamento e Avaliação
    print("\n🚀 Iniciando pipeline de treinamento e avaliação...")
    
    # Avaliação Baseline
    baseline_results = trainer.evaluate(eval_dataset=test_dataset)
    baseline_accuracy = baseline_results['eval_accuracy']
    print(f"\n🎯 Acurácia Baseline (pré-treinado): {baseline_accuracy:.4f}")

    # Treinamento
    trainer.train()
    print("✅ Treinamento concluído.")

    # Avaliação Final
    final_results_dev = trainer.evaluate(eval_dataset=dev_dataset)
    final_results_test = trainer.evaluate(eval_dataset=test_dataset)
    print(f"🎯 Acurácia Final (Dev): {final_results_dev['eval_accuracy']:.4f}")
    print(f"🎯 Acurácia Final (Test): {final_results_test['eval_accuracy']:.4f}")
    
    # --- Geração e Salvamento de Resultados ---
    model_id = os.path.basename(output_dir)
    results_dir = os.path.join("../../results/covid")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n💾 Salvando predições e métricas em {results_dir} com ID: {model_id}")

    # Salvar Modelo primeiro
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"   - Checkpoint do modelo salvo em: {output_dir}")

    # Fazer e salvar predições
    dev_preds_output = trainer.predict(dev_dataset)
    test_preds_output = trainer.predict(test_dataset)

    # O output de predict pode ser uma tupla se o modelo retornar múltiplos outputs.
    # Os logits que queremos são geralmente o primeiro elemento.
    dev_preds_raw = dev_preds_output.predictions
    test_preds_raw = test_preds_output.predictions
    test_labels = test_preds_output.label_ids

    dev_predictions = dev_preds_raw[0] if isinstance(dev_preds_raw, tuple) else dev_preds_raw
    test_predictions = test_preds_raw[0] if isinstance(test_preds_raw, tuple) else test_preds_raw

    dev_predictions_df = pd.DataFrame({
        'sequence': dev_df['sequence'],
        'label': dev_df['label'],
        'prediction': np.argmax(dev_predictions, axis=1)
    })
    test_predictions_df = pd.DataFrame({
        'sequence': test_df['sequence'],
        'label': test_df['label'],
        'prediction': np.argmax(test_predictions, axis=1)
    })

    dev_preds_path = os.path.join(results_dir, f"{model_id}_dev_predictions.csv")
    test_preds_path = os.path.join(results_dir, f"{model_id}_test_predictions.csv")
    dev_predictions_df.to_csv(dev_preds_path, index=False)
    test_predictions_df.to_csv(test_preds_path, index=False)
    print(f"   - Predições salvas em {dev_preds_path} e {test_preds_path}")

    # Consolidar e salvar métricas
    report_dict = classification_report(
        test_labels, np.argmax(test_predictions, axis=1), output_dict=True
    )

    results_summary = {
        'model_id': model_id,
        'profile': profile_name,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
        },
        'training_config': profile,
        'results': {
            'baseline_accuracy': baseline_accuracy,
            'final_accuracy_dev': final_results_dev['eval_accuracy'],
            'final_accuracy_test': final_results_test['eval_accuracy'],
            'classification_report_test': report_dict
        }
    }
    
    results_json_path = os.path.join(results_dir, f"{model_id}_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"   - Métricas salvas em: {results_json_path}")
    print("✅ Processo concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o treinamento do DNABERT-2 para COVID com base em um perfil de configuração.")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Nome do perfil de treinamento a ser usado do arquivo config.yaml."
    )
    args = parser.parse_args()
    main(args.profile) 