import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_covid_data(n_samples=None):
    """Load COVID variant data"""
    test_df = pd.read_csv("GUE/virus/covid/dev.csv")
    
    # Sample data for quick demo
    if n_samples:
        print("Extraindo amostra do conjunto 'dev'")
        test_df = test_df.sample(n=n_samples, random_state=0)
    
    print(f"COVID dataset size: {len(test_df)}")
    print(f"Variant distribution: {test_df['label'].value_counts().sort_index().to_dict()}")
    
    return test_df

def run_pretrained_inference(test_df):
    """Run inference using pre-trained DNABERT-2 without fine-tuning"""
    print("\n=== Pre-trained Model Inference (Expected: Random Performance) ===")
    
    # Load model for 9-class classification (COVID variants 0-8)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M", 
        trust_remote_code=True, 
        num_labels=9
    )
    model.to('cuda')
    
    sequences = test_df['sequence'].tolist()
    labels = test_df['label'].tolist()
    
    model.eval()
    predictions = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, sequence in enumerate(sequences):
            # Truncate very long sequences
            if len(sequence) > 2000:
                sequence = sequence[:2000]
                
            inputs = tokenizer(
                sequence, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors="pt"
            )
            inputs.to('cuda')
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(sequences)} sequences")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    print(f"\nPre-trained Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Expected random accuracy for 9 classes: ~0.111")
    
    # Show confusion between predicted and actual
    variant_names = [f"Variant_{i}" for i in range(9)]
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=variant_names, zero_division=0))
    
    return accuracy

def analyze_sequences(test_df):
    """Analyze sequence characteristics"""
    print("\n=== Sequence Analysis ===")
    sequences = test_df['sequence']
    lengths = [len(seq) for seq in sequences]
    
    print(f"Sequence length stats:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.0f}")
    print(f"  Median: {np.median(lengths):.0f}")
    
    # Sample sequences by variant
    print(f"\nSample sequences by variant:")
    for variant in sorted(test_df['label'].unique()):
        sample_seq = test_df[test_df['label'] == variant]['sequence'].iloc[0]
        print(f"Variant {variant}: {sample_seq[:100]}... (length: {len(sample_seq)})")

def main():
    print("COVID-19 Variant Classification Analysis with DNABERT-2")
    print("=" * 60)
    
    # Load data
    test_df = load_covid_data()  # Small sample for demo
    
    # Analyze sequences
    analyze_sequences(test_df)
    
    # Run pre-trained inference (will show random performance)
    accuracy = run_pretrained_inference(test_df)
    
    print(f"\n=== Summary ===")
    print(f"✓ Dataset: COVID-19 variants (9 classes)")
    print(f"✓ Sequences analyzed: {len(test_df)}")
    print(f"✓ Pre-trained accuracy: {accuracy:.4f}")
    print(f"✓ Note: Low accuracy expected - model needs fine-tuning for COVID classification")
    print(f"\nTo achieve good performance (~85%+ as reported in paper),")
    print(f"the model needs to be fine-tuned on COVID variant training data.")

if __name__ == "__main__":
    main()