#!/usr/bin/env python3
"""
Script para Avaliação e Geração de Embeddings com DNABERT-2 treinado.

Este script carrega um modelo fine-tuned e o utiliza para:
1.  Calcular a acurácia em um conjunto de dados de teste.
2.  Gerar embeddings de sequências de DNA.
3.  Visualizar os embeddings (ex: com t-SNE ou UMAP).

Uso:
    python covid_examples/evaluate.py \
        --model_path ./results/full_training_gpu_optimized/checkpoint-XXXX \
        --test_data GUE/virus/covid/test.csv
"""
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def get_embedding(sequence, tokenizer, model, device='cuda'):
    """Gera o embedding para uma única sequência de DNA."""
    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Usa o embedding do token [CLS] (primeiro token)
        embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    return embedding

def main(args):
    print(f"Carregando modelo de: {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()

    print(f"Carregando dados de teste de: {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    if args.sample_size:
        print(f"Usando uma amostra de {args.sample_size} dados.")
        test_df = test_df.sample(n=args.sample_size, random_state=42)

    sequences = test_df['sequence'].tolist()
    labels = test_df['label'].tolist()
    
    print("Gerando embeddings...")
    embeddings = []
    for seq in tqdm(sequences, desc="Gerando Embeddings"):
        embeddings.append(get_embedding(seq, tokenizer, model, device))
        
    embeddings = np.array(embeddings)
    
    # Salvar embeddings e labels
    output_path_embeddings = f"{args.output_dir}/embeddings.npy"
    output_path_labels = f"{args.output_dir}/labels.npy"
    os.makedirs(args.output_dir, exist_ok=True)
    
    np.save(output_path_embeddings, embeddings)
    np.save(output_path_labels, np.array(labels))

    print(f"\nEmbeddings salvos em: {output_path_embeddings}")
    print(f"Labels salvos em: {output_path_labels}")
    print("\nPróximo passo: Use um script de visualização para plotar os embeddings com t-SNE ou UMAP.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação e Geração de Embeddings COVID-19.")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o checkpoint do modelo treinado.")
    parser.add_argument("--test_data", type=str, default="GUE/virus/covid/test.csv", help="Caminho para os dados de teste.")
    parser.add_argument("--output_dir", type=str, default="./results/embeddings", help="Diretório para salvar os embeddings.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Tamanho da amostra para gerar embeddings.")
    args = parser.parse_args()
    main(args) 