"""
Training Script for Document Classifier and NER Model.
Data preparation, training, and evaluation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw dataset into train/val/test splits."""
    
    @staticmethod
    def load_json_dataset(json_path: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def split_dataset(data: List[Dict], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test.
        
        Returns:
            (train_data, val_data, test_data)
        """
        np.random.seed(seed)
        
        # First split: train + temp
        train_data, temp_data = train_test_split(
            data, test_size=(val_ratio + test_ratio),
            random_state=seed
        )
        
        # Second split: val + test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed
        )
        
        logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    @staticmethod
    def prepare_classifier_data(dataset: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Prepare data for document classifier.
        
        Returns:
            (texts, labels)
        """
        texts = []
        labels = []
        
        for item in dataset:
            texts.append(item.get("raw_text", ""))
            labels.append(item.get("document_type", "unknown"))
        
        return texts, labels
    
    @staticmethod
    def prepare_ner_data(dataset: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Prepare data for NER model.
        
        Returns:
            (tokens_list, tags_list)
        """
        tokens_list = []
        tags_list = []
        
        for item in dataset:
            lines = item.get("lines", [])
            
            for line in lines:
                tokens = line.get("tokens", [])
                tags = line.get("ner_tags", [])
                
                if tokens and tags:
                    tokens_list.append(tokens)
                    tags_list.append(tags)
        
        logger.info(f"Prepared {len(tokens_list)} NER samples")
        return tokens_list, tags_list


class ClassifierTrainer:
    """Train document classifier."""
    
    def __init__(self):
        from src.classification import DocumentClassifier
        self.classifier = DocumentClassifier("logistic_regression", "tfidf")
    
    def train(self, texts: List[str], labels: List[str]):
        """Train classifier."""
        logger.info(f"Training classifier on {len(texts)} samples")
        self.classifier.train(texts, labels)
    
    def evaluate(self, texts: List[str], labels: List[str]):
        """Evaluate classifier."""
        logger.info("Evaluating classifier...")
        metrics = self.classifier.evaluate(texts, labels)
        return metrics
    
    def save(self, model_path: str):
        """Save trained classifier."""
        self.classifier.save(model_path)
        logger.info(f"Classifier saved to {model_path}")


class NERTrainer:
    """Train NER model."""
    
    def __init__(self):
        from config import TAG_TO_ID, ID_TO_TAG
        from src.extraction import MedicalNERModel
        
        self.ner_model = MedicalNERModel(
            num_labels=len(TAG_TO_ID),
            tag_to_id=TAG_TO_ID,
            id_to_tag=ID_TO_TAG,
        )
    
    def train(self,
              train_tokens: List[List[str]],
              train_tags: List[List[str]],
              val_tokens: List[List[str]],
              val_tags: List[List[str]],
              num_epochs: int = 3,
              batch_size: int = 8,
              output_dir: str = "./models/ner_model"):
        """Train NER model."""
        logger.info(f"Training NER model on {len(train_tokens)} samples")
        
        self.ner_model.train(
            train_tokens=train_tokens,
            train_tags=train_tags,
            val_tokens=val_tokens,
            val_tags=val_tags,
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=output_dir,
        )
    
    def evaluate(self, tokens: List[List[str]], tags: List[List[str]]):
        """Simple evaluation on sample predictions."""
        logger.info("Evaluating NER model on sample...")
        
        correct = 0
        total = 0
        
        for token_seq, tag_seq in list(zip(tokens, tags))[:100]:  # Sample 100
            predictions = self.ner_model.predict(token_seq)
            
            for pred, true in zip(predictions, tag_seq):
                if pred == true:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Sample accuracy: {accuracy:.4f}")
        
        return {"accuracy": accuracy}


# =======================
# Main Training Pipeline
# =======================

def train_all_models(dataset_path: str,
                     output_dir: str = "./models",
                     num_epochs: int = 3):
    """
    Complete training pipeline.
    
    Args:
        dataset_path: Path to healthcare_dataset.json
        output_dir: Directory to save models
        num_epochs: Number of training epochs for NER
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("MEDICAL DOCUMENT ANALYZER - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and split dataset
    logger.info("\n[Step 1] Loading dataset...")
    data_processor = DataProcessor()
    dataset = data_processor.load_json_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    train_data, val_data, test_data = data_processor.split_dataset(dataset)
    
    # Step 2: Train Classifier
    logger.info("\n[Step 2] Training Document Classifier...")
    classifier_trainer = ClassifierTrainer()
    
    train_texts, train_labels = data_processor.prepare_classifier_data(train_data)
    val_texts, val_labels = data_processor.prepare_classifier_data(val_data)
    test_texts, test_labels = data_processor.prepare_classifier_data(test_data)
    
    classifier_trainer.train(train_texts, train_labels)
    classifier_metrics = classifier_trainer.evaluate(test_texts, test_labels)
    
    classifier_trainer.save(f"{output_dir}/classifier/classifier.pkl")
    
    logger.info(f"Classifier Test Accuracy: {classifier_metrics['accuracy']:.4f}")
    
    # Step 3: Train NER Model
    logger.info("\n[Step 3] Training NER Model...")
    ner_trainer = NERTrainer()
    
    train_tokens, train_tags = data_processor.prepare_ner_data(train_data)
    val_tokens, val_tags = data_processor.prepare_ner_data(val_data)
    
    ner_trainer.train(
        train_tokens=train_tokens,
        train_tags=train_tags,
        val_tokens=val_tokens,
        val_tags=val_tags,
        num_epochs=num_epochs,
        batch_size=8,
        output_dir=f"{output_dir}/ner_model",
    )
    
    ner_metrics = ner_trainer.evaluate(val_tokens, val_tags)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Classifier Test Accuracy: {classifier_metrics['accuracy']:.4f}")
    logger.info(f"NER Validation Accuracy: {ner_metrics['accuracy']:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ner_trainer.py <dataset_path> [output_dir] [num_epochs]")
        print("Example: python ner_trainer.py healthcare_dataset.json ./models 3")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./models"
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    train_all_models(dataset_path, output_dir, num_epochs)
