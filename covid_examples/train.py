#!/usr/bin/env python3
"""
Script de Treinamento para Classificação de Variantes de COVID-19.

Este script unificado realiza o fine-tuning do modelo DNABERT-2
usando perfis de configuração em formato YAML.

Uso:
  - Para um treino completo e otimizado:
    python covid_examples/train.py --profile covid_examples/profile_full_train.yaml

  - Para um fine-tuning rápido em uma amostra de dados:
    python covid_examples/train.py --profile covid_examples/profile_finetune.yaml
"""
import os
import argparse
import yaml
import torch
import gc
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
from utils import load_and_prepare_data, compute_metrics, generate_final_reports

def main(profile_path):
    # --- 1. Carregar Perfil de Configuração ---
    print(f"Carregando perfil de: {profile_path}")
    with open(profile_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_config = config.get('experiment', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    run_name = exp_config.get('run_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    training_config['output_dir'] = os.path.join(training_config.get('output_dir', './results'), run_name)
    
    print(f"Iniciando experimento: {exp_config.get('description', 'N/A')}")
    print(f"   - Resultados salvos em: {training_config['output_dir']}")

    # --- 2. Setup do Ambiente ---
    torch.cuda.empty_cache()

    # --- 3. Carregar Tokenizer e Modelo ---
    print("\nCarregando DNABERT-2...")
    tokenizer = AutoTokenizer.from_pretrained(model_config['name_or_path'], trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['name_or_path'],
        num_labels=model_config['num_labels'],
        trust_remote_code=True
    )

    # --- 4. Carregar Dados ---
    train_dataset, dev_dataset, test_dataset = load_and_prepare_data(config, tokenizer)

    # --- 5. Configurar e Executar o Treinamento ---
    training_args = TrainingArguments(**training_config)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if config.get('callbacks', {}).get('early_stopping', {}).get('enabled') else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks
    )
    
    # Avaliação de baseline antes do treino
    if exp_config.get('run_baseline_evaluation', False):
        print("\nAvaliando baseline pré-treino...")
        baseline_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="baseline")
        print(f"Acurácia Baseline: {baseline_results['baseline_accuracy']:.4f}")

    # Treinamento
    print("\nIniciando fine-tuning...")
    trainer.train()
    print("Treinamento concluído.")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Modelo salvo em: {training_args.output_dir}")

    # Avaliação final
    print("\nAvaliando modelo final...")
    final_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="final")
    print(f"Acurácia Final: {final_results['final_accuracy']:.4f}")
    
    if exp_config.get('generate_final_reports', False):
        generate_final_reports(trainer, test_dataset, training_args.output_dir)

    print("\nExperimento concluído!")
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento COVID-19 com DNABERT-2.")
    parser.add_argument(
        '--profile',
        type=str,
        required=True,
        help="Caminho para o arquivo de perfil YAML do experimento."
    )
    args = parser.parse_args()
    main(args.profile) 