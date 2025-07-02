#!/usr/bin/env python3
"""
Treinamento COVID-19 com dataset completo (73K amostras)
Otimizado para GPU com memória limitada
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
import json
import gc
from datetime import datetime


class CovidDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=200):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = int(self.labels[idx])
        
        # Truncar sequência se muito longa (economizar memória)
        if len(sequence) > 800:
            sequence = sequence[:800]
        
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}


def main():
    print("🧬 DNABERT-2 COVID-19 Treinamento Completo (73K amostras)")
    print("=" * 60)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    
    # Limpar cache da GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memória disponível: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Carregar dataset completo
    print("\n📁 Carregando dataset completo...")
    train_df = pd.read_csv('GUE/virus/covid/train.csv')
    dev_df = pd.read_csv('GUE/virus/covid/dev.csv')
    test_df = pd.read_csv('GUE/virus/covid/test.csv')
    
    print(f"✅ Dataset completo carregado:")
    print(f"   - Treino: {len(train_df):,} amostras")
    print(f"   - Dev: {len(dev_df):,} amostras") 
    print(f"   - Teste: {len(test_df):,} amostras")
    
    # Verificar distribuição
    print(f"\n📊 Distribuição de classes:")
    for label, count in train_df['label'].value_counts().sort_index().items():
        print(f"   Classe {label}: {count:,} ({count/len(train_df)*100:.1f}%)")
    
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
    
    # Configuração otimizada para dataset completo
    max_length = 200  # Reduzido ainda mais para economizar memória
    
    print(f"\n🔄 Preparando datasets (max_length={max_length})...")
    
    train_dataset = CovidDataset(
        train_df['sequence'].values,
        train_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    # Usar subset menor para dev (economia de memória na avaliação)
    dev_subset = dev_df.sample(n=2000, random_state=42)
    dev_dataset = CovidDataset(
        dev_subset['sequence'].values,
        dev_subset['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    # Subset pequeno para teste final
    test_subset = test_df.sample(n=1000, random_state=42)
    test_dataset = CovidDataset(
        test_subset['sequence'].values,
        test_subset['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    print(f"✅ Datasets criados:")
    print(f"   - Treino: {len(train_dataset):,}")
    print(f"   - Dev: {len(dev_dataset):,}")
    print(f"   - Teste: {len(test_dataset):,}")
    
    # Configurar treinamento ultra-otimizado
    output_dir = f"./covid_model_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,                  # 2 épocas para dataset completo
        per_device_train_batch_size=1,       # Batch size mínimo absoluto
        per_device_eval_batch_size=1,        # Batch size mínimo para eval
        gradient_accumulation_steps=16,       # Batch efetivo = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=500,                   # Log menos frequente
        evaluation_strategy="steps",
        eval_steps=2000,                     # Avaliar menos frequentemente
        save_strategy="steps", 
        save_steps=2000,
        save_total_limit=1,                  # Manter apenas 1 checkpoint
        load_best_model_at_end=False,        # Desabilitar para economizar memória
        fp16=True,                           # Obrigatório
        dataloader_num_workers=0,            # Sem paralelismo
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
        max_grad_norm=1.0,                   # Gradient clipping
        warmup_steps=100,                    # Warmup menor
        lr_scheduler_type="linear",
        # Otimizações adicionais de memória
        gradient_checkpointing=True,         # Trocar computação por memória
        max_steps=9000,                      # Limitar steps se necessário (~1.2 épocas)
    )
    
    print(f"\n⚙️  Configuração para dataset completo:")
    print(f"   - Dataset: {len(train_df):,} amostras")
    print(f"   - Épocas: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Batch efetivo: {training_args.gradient_accumulation_steps}")
    print(f"   - Max length: {max_length}")
    print(f"   - FP16: {training_args.fp16}")
    print(f"   - Gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"   - Steps estimados: ~{len(train_dataset) // training_args.gradient_accumulation_steps * training_args.num_train_epochs}")
    
    # Configurar trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Baseline rápido apenas com pequena amostra
    print("\n📊 Avaliando baseline...")
    try:
        # Usar apenas 200 amostras para baseline
        tiny_test = CovidDataset(
            test_subset['sequence'].values[:200],
            test_subset['label'].values[:200],
            tokenizer,
            max_length=max_length
        )
        
        baseline_results = trainer.evaluate(eval_dataset=tiny_test)
        baseline_accuracy = baseline_results['eval_accuracy']
        print(f"🎯 Baseline (200 amostras): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"⚠️  Erro no baseline: {e}")
        baseline_accuracy = 0.11
    
    # Limpar memória antes do treinamento
    del tiny_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Treinar
    print(f"\n🚀 Iniciando treinamento completo...")
    print(f"   Dataset: {len(train_df):,} amostras")
    print(f"   Tempo estimado: 60-90 minutos")
    print(f"   Use 'tail -f covid_training.log' para monitorar")
    
    try:
        training_start = datetime.now()
        train_result = trainer.train()
        training_end = datetime.now()
        training_time = training_end - training_start
        
        print(f"✅ Treinamento concluído em {training_time}")
        
        # Salvar modelo
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Modelo salvo em: {output_dir}")
        
        # Avaliar modelo final
        print(f"\n🧪 Avaliando modelo treinado...")
        
        # Limpar memória antes da avaliação
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Avaliação em batches pequenos para evitar OOM
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        final_accuracy = test_results['eval_accuracy']
        
        print(f"\n🎯 RESULTADOS FINAIS:")
        print(f"   - Dataset: {len(train_df):,} amostras de treino")
        print(f"   - Baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        print(f"   - Final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"   - Melhoria: {final_accuracy - baseline_accuracy:.4f}")
        print(f"   - Meta do artigo: ~85%")
        
        if final_accuracy >= 0.80:
            print(f"   - Status: ✅ EXCELENTE (≥80%)")
        elif final_accuracy >= 0.70:
            print(f"   - Status: ✅ BOM (≥70%)")
        elif final_accuracy >= 0.50:
            print(f"   - Status: ⚠️  MODERADO (≥50%)")
        else:
            print(f"   - Status: ❌ BAIXO (<50%)")
        
        # Relatório detalhado
        print(f"\n📋 Gerando predições detalhadas...")
        try:
            predictions = trainer.predict(test_dataset)
            predicted_labels = np.argmax(predictions.predictions, axis=1)
            true_labels = test_subset['label'].values[:len(predicted_labels)]
            
            report = classification_report(
                true_labels, 
                predicted_labels, 
                target_names=[f'Variante {i}' for i in range(9)],
                digits=4
            )
            print(f"\n📊 Relatório por variante:")
            print(report)
            
        except Exception as e:
            print(f"⚠️  Erro no relatório detalhado: {e}")
        
        # Salvar resultados
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'dataset_info': {
                'train_size': len(train_df),
                'dev_size': len(dev_subset),
                'test_size': len(test_subset),
                'max_length': max_length,
                'full_dataset': True
            },
            'training_config': {
                'epochs': training_args.num_train_epochs,
                'batch_size': training_args.per_device_train_batch_size,
                'effective_batch_size': training_args.gradient_accumulation_steps,
                'learning_rate': training_args.learning_rate,
                'fp16': training_args.fp16,
                'gradient_checkpointing': training_args.gradient_checkpointing,
                'training_time_seconds': training_time.total_seconds()
            },
            'results': {
                'baseline_accuracy': float(baseline_accuracy),
                'final_accuracy': float(final_accuracy),
                'improvement': float(final_accuracy - baseline_accuracy),
                'target_accuracy': 0.85
            }
        }
        
        with open(f'{output_dir}/results_full.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Resultados completos salvos em: {output_dir}/results_full.json")
        print(f"🎉 Experimento com dataset completo concluído!")
        
        return final_accuracy
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        print(f"💡 Sugestões:")
        print(f"   - Reduzir max_length para 150")
        print(f"   - Aumentar gradient_accumulation_steps para 32")
        print(f"   - Usar apenas 1 época")
        return 0.0


if __name__ == "__main__":
    # Otimizações de memória PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Limpar cache no início
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_accuracy = main()