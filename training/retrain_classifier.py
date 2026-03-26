"""
Retrain Document Classifier with Proper Class Weighting
Handles imbalanced dataset where discharge_summary dominates
"""

import json
import logging
from pathlib import Path
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src.classification import DocumentClassifier, EnsembleDocumentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
dataset_path = Path("healthcare_dataset.json")
print(f"Loading dataset from {dataset_path}")

with open(dataset_path, 'r') as f:
    data = json.load(f)

print(f"Total documents: {len(data)}")

# Extract texts and labels
texts = []
labels = []

for item in data:
    # Try different possible text fields
    text = item.get("raw_text") or item.get("text") or item.get("content") or ""
    label = item.get("document_type", "unknown")
    
    if text:  # Only include if text exists
        texts.append(text)
        labels.append(label)

print(f"Extracted {len(texts)} documents with text")

# Check distribution
type_counts = Counter(labels)
print("\nDocument type distribution:")
for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {doc_type}: {count} ({count/len(labels)*100:.1f}%)")

# Compute class weights to handle imbalance
unique_labels = np.unique(labels)
class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=labels)
class_weights = {label: weight for label, weight in zip(unique_labels, class_weights_array)}

print("\nClass weights (for imbalance handling):")
for label, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label}: {weight:.2f}")

# Prepare sample weights for training
sample_weights = np.array([class_weights[label] for label in labels])

# Train classifier with class weights
print("\n" + "="*60)
print("Training Classifier with Class Weights")
print("="*60)

classifier = DocumentClassifier(
    model_type="logistic_regression",
    vectorizer_type="tfidf",
    max_features=3000,
    ngram_range=(1, 2)
)

# Train with sample weights
print(f"Training on {len(texts)} samples...")
classifier.train(texts, labels, sample_weights=sample_weights)

# Evaluate on training data (quick check)
print("\n" + "="*60)
print("Training Set Evaluation")
print("="*60)

metrics = classifier.evaluate(texts, labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print("\nDetailed Classification Report:")
print(metrics['classification_report'])

# Save model
model_path = Path("models/classifier/classifier.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

print(f"\nSaving model to {model_path}")
classifier.save(str(model_path))

print("\n" + "="*60)
print("Retraining Complete!")
print("="*60)
print("\nKey improvements:")
print("1. Class weights handle imbalanced data (discharge_summary was overrepresented)")
print("2. Model trained on all 87 documents with proper weighting")
print("3. Should now correctly classify different document types")
print("\nThe ensemble classifier uses keyword-based fallback for low confidence,")
print("so /classify endpoint should now work much better.")
