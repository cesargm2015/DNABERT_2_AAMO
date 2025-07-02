import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações ---
# O modelo agora será salvo em 'checkpoints', então o caminho precisa refletir isso.
# Este é um exemplo, ajuste para o diretório do modelo que você quer visualizar.
MODEL_PATH = "../../checkpoints/low_vram_finetune_20250701_221539/" # Modelo mais recente disponível
DATA_PATH = "../../GUE/virus/covid/dev.csv" # Ajustado para a nova localização do script
OUTPUT_DIR = "embedding_visualizations"
SAMPLE_FRACTION = 0.5 

# --- Criar diretório de saída se não existir ---

# --- Criar diretório de saída se não existir ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Carregar Modelo e Tokenizador ---
print("Carregando modelo e tokenizador...")
config = AutoConfig.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Modelo carregado com sucesso.")

# --- Carregar e Amostrar os Dados ---
print(f"Carregando dados de {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(frac=SAMPLE_FRACTION, random_state=42)
sequences = df_sample['sequence'].tolist()
labels = df_sample['label'].tolist()
print(f"{len(df_sample)} amostras carregadas.")

# --- Gerar Embeddings ---
def get_embeddings(sequences, tokenizer, model, device):
    print("Gerando embeddings...")
    all_embeddings = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # Usar o last_hidden_state (primeiro elemento da tupla)
            last_hidden_state = outputs[0]
            
            # Usar o embedding do token [CLS] (primeiro token) do último estado oculto
            cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding)
            
    print("Embeddings gerados.")
    return np.vstack(all_embeddings)

embeddings = get_embeddings(sequences, tokenizer, model, device)

# --- Redução de Dimensionalidade e Visualização ---
def plot_projection(embeddings_2d, labels, title, filename):
    print(f"Gerando gráfico: {title}")
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("Set3", n_colors=len(set(labels)))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        s=50 
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Componente 1", fontsize=12)
    plt.ylabel("Componente 2", fontsize=12)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Gráfico salvo em {os.path.join(OUTPUT_DIR, filename)}")

# PCA
print("Aplicando PCA...")
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)
plot_projection(embeddings_pca, labels, "Projeção 2D com PCA", "pca_projection.png")

# t-SNE
print("Aplicando t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_sample)-1))
embeddings_tsne = tsne.fit_transform(embeddings)
plot_projection(embeddings_tsne, labels, "Projeção 2D com t-SNE", "tsne_projection.png")

# UMAP
print("Aplicando UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_umap = reducer.fit_transform(embeddings)
plot_projection(embeddings_umap, labels, "Projeção 2D com UMAP", "umap_projection.png")

print("Processo concluído.") 