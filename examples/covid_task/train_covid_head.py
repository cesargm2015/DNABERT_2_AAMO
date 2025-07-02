#!/usr/bin/env python3
"""
Treinamento do cabeçalho de classificação para variantes COVID-19 com DNABERT-2.
Os pesos do transformer DNABERT-2 são congelados.

Meta: Treinar um classificador leve sobre os embeddings do DNABERT-2.
Dataset: 73K treino + 9K dev + 9K teste
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import os
import json
from datetime import datetime
from transformers.trainer_pt_utils import nested_detach


class CovidDataset(Dataset):
    """Dataset customizado para sequências COVID-19"""
    
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
        
        # Tokenizar sequência
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


def load_covid_data():
    """Carrega dados COVID-19 do dataset GUE"""
    print("📁 Carregando dataset COVID-19...")
    
    train_df = pd.read_csv('GUE/virus/covid/train.csv')
    dev_df = pd.read_csv('GUE/virus/covid/dev.csv')
    test_df = pd.read_csv('GUE/virus/covid/test.csv')
    
    print(f"✅ Dados carregados:")
    print(f"   - Treino: {len(train_df):,} amostras")
    print(f"   - Dev: {len(dev_df):,} amostras")
    print(f"   - Teste: {len(test_df):,} amostras")
    
    # Verificar distribuição de classes
    print(f"\n📊 Distribuição de classes (treino):")
    train_class_dist = train_df['label'].value_counts().sort_index()
    for label, count in train_class_dist.items():
        print(f"   Classe {label}: {count:,} ({count/len(train_df)*100:.1f}%)")
    
    # Verificar comprimento das sequências
    seq_lengths = train_df['sequence'].str.len()
    print(f"\n📏 Comprimento das sequências:")
    print(f"   - Mínimo: {seq_lengths.min()}")
    print(f"   - Máximo: {seq_lengths.max()}")
    print(f"   - Médio: {seq_lengths.mean():.1f}")
    print(f"   - Mediano: {seq_lengths.median()}")
    
    return train_df, dev_df, test_df


def compute_metrics(eval_pred):
    """Função para calcular métricas durante o treinamento"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
    }


def memory_efficient_evaluate(model, tokenizer, dataset, batch_size):
    """
    Avaliação com uso eficiente de memória, processando em batches.
    """
    model.eval()
    all_preds = []
    all_labels = []

    data_loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )

    for step, inputs in enumerate(data_loader):
        # Mover inputs para o dispositivo correto
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        labels = inputs["labels"]

        # Mover predições e rótulos para a CPU para economizar memória da GPU
        all_preds.append(nested_detach(logits).cpu())
        all_labels.append(nested_detach(labels).cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calcular métricas
    predictions = np.argmax(all_preds.numpy(), axis=1)
    accuracy = accuracy_score(all_labels.numpy(), predictions)
    
    return {'eval_accuracy': accuracy}


class MemoryEfficientTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        
        # Nossa função já retorna a chave correta ('eval_accuracy')
        # então não precisamos adicionar o prefixo novamente.
        metrics = memory_efficient_evaluate(
            self.model, self.tokenizer, eval_dataset, self.args.per_device_eval_batch_size
        )

        # Apenas logamos as métricas
        self.log(metrics)

        return metrics


def main():
    """Pipeline de treinamento do cabeçalho de classificação COVID-19"""
    
    print("🧬 DNABERT-2 COVID-19 Treinamento de Cabeçalho")
    print("=" * 50)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memória: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Carregar dados
    train_df, dev_df, test_df = load_covid_data()
    
    # Inicializar tokenizer e modelo
    print("\n🤖 Carregando DNABERT-2...")
    model_name = "zhihan1996/DNABERT-2-117M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=9  # 9 variantes COVID (0-8)
    )

    # ==================================================================
    # === Congelar os pesos do modelo base (DNABERT-2) ===
    # ==================================================================
    print("🧊 Congelando os pesos do DNABERT-2 (treinando apenas o cabeçalho)...")
    total_params = sum(p.numel() for p in model.parameters())
    
    for param in model.bert.parameters():
        param.requires_grad = False
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - Parâmetros totais: {total_params:,}")
    print(f"   - Parâmetros treináveis: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    # ==================================================================

    model.to('cuda')
    
    print(f"✅ Modelo carregado: {model_name}")
    
    # Testar tokenização
    sample_seq = train_df['sequence'].iloc[0]
    sample_tokens = tokenizer(sample_seq, return_tensors='pt')
    print(f"📝 Teste de tokenização:")
    print(f"   Sequência original: {len(sample_seq)} caracteres")
    print(f"   Tokens: {sample_tokens['input_ids'].shape[1]} tokens")
    
    # Criar datasets
    print("\n🔄 Preparando datasets...")
    
    # Determinar max_length baseado nos dados
    max_length = 512  # Conforme relatório
    
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
    
    # Configurar argumentos de treinamento
    output_dir = f"./covid_model_head_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,             # Aumentado, pois treinamos menos parâmetros
        per_device_train_batch_size=8,  # Pode ser maior, pois o gradiente é menor
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  
        learning_rate=1e-3,             # Taxa maior é geralmente melhor para cabeçalhos
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
    )
    
    print(f"\n⚙️  Configuração de treinamento:")
    print(f"   - Épocas: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Max length: {max_length}")
    print(f"   - Diretório de saída: {output_dir}")
    
    # Configurar trainer customizado
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], # Aumentada a paciência
        tokenizer=tokenizer  # Passar o tokenizer para o trainer customizado
    )
    
    # Avaliar modelo pré-treinado (baseline)
    print("\n📊 Avaliando modelo pré-treinado (baseline)...")
    
    # Agora o trainer.evaluate() usa nossa função customizada
    baseline_results = trainer.evaluate(eval_dataset=test_dataset)
    baseline_accuracy = baseline_results['eval_accuracy']
    print(f"🎯 Acurácia baseline (pré-treinado): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # Treinar modelo
    print("\n🚀 Iniciando treinamento do cabeçalho...")
    
    training_start = datetime.now()
    train_result = trainer.train()
    training_end = datetime.now()
    training_time = training_end - training_start
    
    print(f"✅ Treinamento concluído em {training_time}")
    
    # Salvar modelo final
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Avaliar no conjunto de teste
    print("\n🧪 Avaliando modelo no conjunto de teste...")
    
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    final_accuracy = test_results['eval_accuracy']
    
    print(f"\n🎯 RESULTADOS FINAIS:")
    print(f"   - Acurácia baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"   - Acurácia final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"   - Melhoria: {final_accuracy - baseline_accuracy:.4f} ({(final_accuracy - baseline_accuracy)*100:.2f} pontos percentuais)")
    
    # Predições detalhadas para análise
    print("\n📋 Gerando relatório detalhado...")
    
    # Para o relatório detalhado, precisaremos das predições completas.
    # Se ainda houver problemas de memória, podemos implementar uma predição em lote aqui também.
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_df['label'].values
    
    # Relatório de classificação
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=[f'Variante {i}' for i in range(9)],
        digits=4
    )
    
    print("\n📊 Relatório de classificação por variante:")
    print(report)
    
    # Matriz de confusão
    cm = confusion_matrix(true_labels, predicted_labels)
    print("\n🔀 Matriz de confusão:")
    print("   (linhas=real, colunas=predito)")
    print(cm)
    
    # Salvar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'training_type': 'head_only',
        'dataset_info': {
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
            'num_classes': 9,
            'max_length': max_length
        },
        'training_config': {
            'epochs': training_args.num_train_epochs,
            'batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
            'training_time_seconds': training_time.total_seconds()
        },
        'results': {
            'baseline_accuracy': float(baseline_accuracy),
            'final_accuracy': float(final_accuracy),
            'improvement': float(final_accuracy - baseline_accuracy),
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    results_file = f'{output_dir}/results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {results_file}")
    print(f"📁 Modelo salvo em: {output_dir}")
    
    print(f"\n🎉 Experimento concluído com sucesso!")
    print(f"   Acurácia final: {final_accuracy*100:.2f}%")
    
    return final_accuracy


if __name__ == "__main__":
    final_accuracy = main() 