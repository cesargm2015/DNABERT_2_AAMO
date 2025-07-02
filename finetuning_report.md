# Relatório de Fine-tuning do Modelo DNABERT-2

Este documento resume os resultados do último processo de fine-tuning do modelo DNABERT-2 para a tarefa de classificação de sequências de COVID.

## 📝 Visão Geral

- **Modelo Base:** `zhihan1996/DNABERT-2-117M`
- **Data e Hora:** `2025-07-01T19:34:30.830954`

## 📊 Dataset

| Parâmetro      | Valor |
|----------------|-------|
| Tamanho (Treino) | 500   |
| Tamanho (Dev)    | 500   |
| Tamanho (Teste)  | 500   |
| Nº de Classes  | 9     |
| Tam. Máx. Seq. | 512   |

## ⚙️ Hiperparâmetros de Treinamento

| Parâmetro       | Valor            |
|-----------------|------------------|
| Épocas          | 1                |
| Batch Size      | 1                |
| Taxa de Aprend. | 2e-05            |
| Duração (sec)   | 40.68            |

## 📈 Resultados

| Métrica               | Valor                |
|-----------------------|----------------------|
| Acurácia (Baseline)   | 10.2%                |
| Acurácia (Final)      | 13.6%                |
| Melhoria              | 3.4%                 |
| Acurácia Alvo         | 85.0%                |

## 📋 Relatório de Classificação

```
              precision    recall  f1-score   support

  Variante 0       0.18      0.18      0.18        67
  Variante 1       0.00      0.00      0.00        61
  Variante 2       0.12      0.85      0.22        55
  Variante 3       0.20      0.10      0.14        58
  Variante 4       0.00      0.00      0.00        63
  Variante 5       0.00      0.00      0.00        42
  Variante 6       0.20      0.05      0.08        59
  Variante 7       0.00      0.00      0.00        45
  Variante 8       0.00      0.00      0.00        50

    accuracy                           0.14       500
   macro avg       0.08      0.13      0.07       500
weighted avg       0.09      0.14      0.07       500
``` 