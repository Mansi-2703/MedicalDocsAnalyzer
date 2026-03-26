"""
NER Model using Hugging Face Transformers.
Fine-tuned for medical document IE with BIO tags.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MedicalNERDataset(Dataset):
    """Dataset for NER training compatible with Hugging Face."""
    
    def __init__(self, tokens_list: List[List[str]], tags_list: List[List[str]], 
                 tokenizer, tag_to_id: Dict[str, int], max_len: int = 512):
        """
        Initialize dataset.
        
        Args:
            tokens_list: List of token sequences
            tags_list: List of tag sequences (BIO format)
            tokenizer: Hugging Face tokenizer
            tag_to_id: Dict mapping tag names to IDs
            max_len: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len
        self.encodings = []
        
        # Encode all samples
        for tokens, tags in zip(tokens_list, tags_list):
            self._encode_sample(tokens, tags)
    
    def _encode_sample(self, tokens: List[str], tags: List[str]):
        """Encode a single sample with alignment to subword tokens."""
        # Tokenize and align
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors=None,
        )
        
        # Align labels to subword tokens
        labels = []
        for i, label in enumerate(tags):
            label_id = self.tag_to_id.get(label, self.tag_to_id.get('O', 0))
            labels.append(label_id)
        
        # Pad labels
        if len(labels) < self.max_len:
            labels.extend([-100] * (self.max_len - len(labels)))
        else:
            labels = labels[:self.max_len]
        
        encoding['labels'] = labels
        self.encodings.append(encoding)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        return {key: torch.tensor(val) for key, val in encoding.items()}


class MedicalNERModel:
    """Fine-tuned NER model for medical documents."""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 num_labels: int = 30,
                 tag_to_id: Dict[str, int] = None,
                 id_to_tag: Dict[int, str] = None):
        """
        Initialize NER model.
        
        Args:
            model_name: Hugging Face model name
            num_labels: Number of label classes
            tag_to_id: Tag to ID mapping
            id_to_tag: ID to tag mapping
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tag_to_id = tag_to_id or {}
        self.id_to_tag = id_to_tag or {}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_trained = False
    
    def train(self, 
              train_tokens: List[List[str]], 
              train_tags: List[List[str]],
              val_tokens: Optional[List[List[str]]] = None,
              val_tags: Optional[List[List[str]]] = None,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 5e-5,
              output_dir: str = "./models/ner_model"):
        """
        Fine-tune NER model.
        
        Args:
            train_tokens: Training token sequences
            train_tags: Training tag sequences
            val_tokens: Validation token sequences
            val_tags: Validation tag sequences
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save model
        """
        logger.info(f"Starting NER training with {len(train_tokens)} samples")
        
        # Create datasets
        train_dataset = MedicalNERDataset(
            train_tokens, train_tags, self.tokenizer, self.tag_to_id
        )
        
        eval_dataset = None
        if val_tokens is not None:
            eval_dataset = MedicalNERDataset(
                val_tokens, val_tags, self.tokenizer, self.tag_to_id
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_steps=10,
            save_steps=50,
            eval_strategy="epoch" if eval_dataset else "no",
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            seed=42,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        self.is_trained = True
    
    def predict(self, tokens: List[str]) -> List[str]:
        """
        Predict NER tags for tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of predicted tags
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        )
        
        # Get word_ids before converting to device dict
        word_ids = encoding.word_ids(batch_index=0)
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        # Get predictions
        pred_ids = torch.argmax(logits, dim=2)[0].tolist()
        
        # Remove padding
        predictions = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id >= len(tokens):
                continue
            pred_id = pred_ids[idx]
            predictions.append(self.id_to_tag.get(pred_id, 'O'))
        
        return predictions[:len(tokens)]
    
    def predict_batch(self, tokens_list: List[List[str]]) -> List[List[str]]:
        """Predict for multiple samples."""
        return [self.predict(tokens) for tokens in tokens_list]
    
    def predict_with_confidence(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """
        Predict NER tags with confidence scores.
        
        Returns:
            List of (tag, confidence) tuples
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        )
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        # Get predictions with confidence
        probs = torch.softmax(logits, dim=-1)[0]
        pred_ids = torch.argmax(probs, dim=1).tolist()
        pred_confs = torch.max(probs, dim=1).values.tolist()
        
        # Map to tags
        word_ids = encoding.word_ids()
        predictions = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id >= len(tokens):
                continue
            pred_id = pred_ids[idx]
            confidence = pred_confs[idx]
            tag = self.id_to_tag.get(pred_id, 'O')
            predictions.append((tag, confidence))
        
        return predictions[:len(tokens)]
    
    def save(self, model_path: str):
        """Save model and tokenizer."""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """Load model and tokenizer."""
        model_path = str(model_path)  # Convert Path object to string if needed
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")


# =======================
# Utilities
# =======================

def align_tokens_with_spans(tokens: List[str], 
                           predicted_tags: List[str]) -> List[Dict]:
    """
    Align predicted tags with token spans.
    
    Returns:
        List of {token, tag, entity_type, is_start}
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, predicted_tags):
        if tag == 'O':
            current_entity = None
        elif tag.startswith('B-'):
            # New entity starts
            entity_type = tag[2:]
            current_entity = {
                "tokens": [token],
                "tag": tag,
                "entity_type": entity_type,
            }
            entities.append(current_entity)
        elif tag.startswith('I-'):
            # Continue entity
            entity_type = tag[2:]
            if current_entity and current_entity["entity_type"] == entity_type:
                current_entity["tokens"].append(token)
            else:
                # Start new if entity type changed
                current_entity = {
                    "tokens": [token],
                    "tag": 'B-' + entity_type,
                    "entity_type": entity_type,
                }
                entities.append(current_entity)
    
    return entities


def extract_entities(tokens: List[str], 
                    predicted_tags: List[str]) -> Dict[str, List[str]]:
    """
    Extract named entities grouped by type.
    
    Returns:
        Dict of {entity_type: [values]}
    """
    entities_by_type = {}
    current_entity = None
    
    for token, tag in zip(tokens, predicted_tags):
        if tag == 'O':
            if current_entity:
                entity_type = current_entity["entity_type"]
                value = " ".join(current_entity["tokens"])
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(value)
            current_entity = None
        elif tag.startswith('B-'):
            # Save previous entity
            if current_entity:
                entity_type = current_entity["entity_type"]
                value = " ".join(current_entity["tokens"])
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(value)
            
            # Start new entity
            entity_type = tag[2:]
            current_entity = {
                "tokens": [token],
                "entity_type": entity_type,
            }
        elif tag.startswith('I-'):
            entity_type = tag[2:]
            if current_entity and current_entity["entity_type"] == entity_type:
                current_entity["tokens"].append(token)
    
    # Add last entity
    if current_entity:
        entity_type = current_entity["entity_type"]
        value = " ".join(current_entity["tokens"])
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(value)
    
    return entities_by_type


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example
    from config import TAG_TO_ID, ID_TO_TAG
    
    model = MedicalNERModel(
        model_name="microsoft/deberta-v3-small",
        num_labels=len(TAG_TO_ID),
        tag_to_id=TAG_TO_ID,
        id_to_tag=ID_TO_TAG,
    )
    
    # Example prediction
    tokens = ["Hemoglobin", ":", "13.5", "g/dL"]
    predictions = model.predict(tokens)
    print(f"Predictions: {predictions}")
