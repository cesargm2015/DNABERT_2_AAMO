#!/usr/bin/env python3
"""
Versão simplificada do treinamento COVID-19 para GPU com memória limitada
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import json
from datetime import datetime


class CovidDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=256):
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}


def main():
    print("🧬 DNABERT-2 COVID-19 Treinamento Simplificado")
    print("=" * 50)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    
    # Carregar subset dos dados para teste
    print("📁 Carregando subset do dataset...")
    train_df = pd.read_csv('GUE/virus/covid/train.csv').head(5000)  # Usar apenas 5K amostras
    dev_df = pd.read_csv('GUE/virus/covid/dev.csv').head(1000)    # Usar apenas 1K para dev
    test_df = pd.read_csv('GUE/virus/covid/test.csv').head(1000)   # Usar apenas 1K para teste
    
    print(f"✅ Subset carregado:")
    print(f"   - Treino: {len(train_df):,} amostras")
    print(f"   - Dev: {len(dev_df):,} amostras") 
    print(f"   - Teste: {len(test_df):,} amostras")
    
    # Verificar distribuição
    print(f"\n📊 Distribuição de classes:")
    for label, count in train_df['label'].value_counts().sort_index().items():
        print(f"   Classe {label}: {count}")
    
    # Carregar modelo
    print("\n🤖 Carregando DNABERT-2...")
    model_name = "zhihan1996/DNABERT-2-117M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=9
    )
    
    print(f"✅ Modelo carregado")
    
    # Criar datasets com max_length reduzido
    max_length = 256  # Reduzido de 512 para economizar memória
    
    train_dataset = CovidDataset(
        train_df['sequence'].values,
        train_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    dev_dataset = CovidDataset(
        dev_df['sequence'].values,
        dev_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    test_dataset = CovidDataset(
        test_df['sequence'].values,
        test_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    print(f"✅ Datasets criados com max_length={max_length}")
    
    # Configurar treinamento para GPU limitada
    output_dir = f"./covid_model_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,        # Apenas 1 época para teste
        per_device_train_batch_size=1,  # Batch size mínimo
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # Batch efetivo = 8
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",     # Avaliar apenas no final
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,    # Desabilitar para economizar memória
        fp16=True,                       # Obrigatório para economizar memória
        dataloader_num_workers=0,        # Sem paralelismo
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,       # Processar avaliação em steps pequenos
    )
    
    print(f"\n⚙️  Configuração otimizada:")
    print(f"   - Épocas: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Batch efetivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   - Max length: {max_length}")
    print(f"   - FP16: {training_args.fp16}")
    
    # Configurar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Fazer baseline simples apenas com uma pequena amostra
    print("\n📊 Testando modelo pré-treinado...")
    
    # Pegar apenas 100 amostras para baseline
    small_test = CovidDataset(
        test_df['sequence'].values[:100],
        test_df['label'].values[:100],
        tokenizer,
        max_length=max_length
    )
    
    try:
        baseline_results = trainer.evaluate(eval_dataset=small_test)
        baseline_accuracy = baseline_results['eval_accuracy']
        print(f"🎯 Baseline (100 amostras): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"⚠️  Erro no baseline: {e}")
        baseline_accuracy = 0.11  # Aproximadamente baseline aleatória
    
    # Treinar
    print("\n🚀 Iniciando treinamento...")
    
    try:
        training_start = datetime.now()
        train_result = trainer.train()
        training_end = datetime.now()
        training_time = training_end - training_start
        
        print(f"✅ Treinamento concluído em {training_time}")
        
        # Salvar modelo
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Avaliar
        print("\n🧪 Avaliando modelo treinado...")
        test_results = trainer.evaluate(eval_dataset=small_test)
        final_accuracy = test_results['eval_accuracy']
        
        print(f"\n🎯 RESULTADOS (Subset de Teste):")
        print(f"   - Baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        print(f"   - Final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"   - Melhoria: {final_accuracy - baseline_accuracy:.4f}")
        print(f"   - Status: {'✅ MELHOROU' if final_accuracy > baseline_accuracy else '❌ NÃO MELHOROU'}")
        
        # Salvar resultados
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'dataset_info': {
                'train_size': len(train_df),
                'dev_size': len(dev_df),
                'test_size': 100,  # Amostra pequena
                'max_length': max_length
            },
            'training_config': {
                'epochs': training_args.num_train_epochs,
                'batch_size': training_args.per_device_train_batch_size,
                'effective_batch_size': training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                'learning_rate': training_args.learning_rate,
                'fp16': training_args.fp16,
                'training_time_seconds': training_time.total_seconds()
            },
            'results': {
                'baseline_accuracy': float(baseline_accuracy),
                'final_accuracy': float(final_accuracy),
                'improvement': float(final_accuracy - baseline_accuracy),
                'note': 'Treinamento em subset devido a limitações de memória GPU'
            }
        }
        
        with open(f'{output_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Resultados salvos em: {output_dir}")
        print(f"🎉 Experimento concluído!")
        
        return final_accuracy
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        return 0.0


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Limpar cache da GPU
    final_accuracy = main()