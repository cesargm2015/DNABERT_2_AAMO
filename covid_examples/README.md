# Exemplo de Classificação de Variantes de COVID-19 com DNABERT-2

Este diretório contém um exemplo completo para treinar, avaliar e utilizar um modelo DNABERT-2 para classificar variantes de COVID-19.

## Estrutura do Diretório

- `train.py`: Script para treinar o modelo.
- `evaluate.py`: Script para avaliar um modelo treinado e gerar embeddings.
- `inference.py`: Script para realizar inferência em novas sequências de DNA.
- `visualize.py`: Script para visualizar os embeddings gerados.
- `utils.py`: Funções de suporte para manipulação de dados e métricas.
- `profile_full_train.yaml`: Perfil de configuração para um treinamento completo e otimizado.
- `profile_finetune.yaml`: Perfil de configuração para um fine-tuning rápido em uma amostra de dados.

## Instruções de Uso

### 1. Treinamento

O script `train.py` utiliza perfis de configuração para definir os parâmetros do treinamento.

**Opção A: Treinamento Completo**

Utiliza o dataset completo (73k amostras) e é otimizado para GPUs com memória limitada.

```bash
python covid_examples/train.py --profile covid_examples/profile_full_train.yaml
```

**Opção B: Fine-tuning Rápido**

Utiliza uma pequena amostra dos dados para um ciclo de desenvolvimento e teste rápido.

```bash
python covid_examples/train.py --profile covid_examples/profile_finetune.yaml
```

Os modelos treinados e os checkpoints serão salvos no diretório `results/` por padrão.

### 2. Avaliação e Geração de Embeddings

Após o treinamento, utilize o script `evaluate.py` para gerar embeddings das sequências de teste usando o modelo treinado.

```bash
python covid_examples/evaluate.py \
    --model_path ./results/nome_do_experimento/checkpoint-XXXX \
    --output_dir ./results/embeddings_gerados
```

- `--model_path`: Caminho para o checkpoint do modelo salvo pelo `Trainer`.
- `--output_dir`: Diretório onde os arquivos `embeddings.npy` e `labels.npy` serão salvos.

### 3. Inferência

Use o script `inference.py` para classificar uma nova sequência de DNA.

```bash
python covid_examples/inference.py \
    --model_path ./results/nome_do_experimento/checkpoint-XXXX \
    --sequence "GATTACA..."
```

### 4. Visualização

Após gerar os embeddings, utilize `visualize.py` para criar uma visualização 2D (t-SNE).

```bash
python covid_examples/visualize.py \
    --embeddings_path ./results/embeddings_gerados/embeddings.npy \
    --labels_path ./results/embeddings_gerados/labels.npy \
    --output_file ./results/visualization.png
``` 