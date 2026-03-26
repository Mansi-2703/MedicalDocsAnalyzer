"""
Document Type Classifier.
Classifies medical documents into types (CBC, LFT, prescription, etc.)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Classify medical documents into document types.
    Trained on TF-IDF or Count vectorization + ML classifier.
    """
    
    def __init__(self, 
                 model_type: str = "logistic_regression",
                 vectorizer_type: str = "tfidf",
                 max_features: int = 3000,
                 ngram_range: Tuple = (1, 2)):
        """
        Initialize classifier.
        
        Args:
            model_type: "logistic_regression", "random_forest", or "svm"
            vectorizer_type: "tfidf" or "count"
            max_features: Max TF-IDF features
            ngram_range: N-gram range (1, 1) for words only, (1, 2) for word+bigrams
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        
        # Build vectorizer
        vectorizer_class = TfidfVectorizer if vectorizer_type == "tfidf" else CountVectorizer
        self.vectorizer = vectorizer_class(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )
        
        # Build classifier
        if model_type == "logistic_regression":
            classifier = LogisticRegression(max_iter=200, random_state=42, multi_class='multinomial')
        elif model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "svm":
            classifier = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.pipe = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', classifier)
        ])
        
        self.classes_ = None
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[str], sample_weights=None):
        """
        Train classifier.
        
        Args:
            texts: List of document texts
            labels: List of document type labels
            sample_weights: Optional sample weights for imbalanced data
        """
        logger.info(f"Training {self.model_type} classifier...")
        logger.info(f"Samples: {len(texts)}, Classes: {len(set(labels))}")
        
        self.pipe.fit(texts, labels, classifier__sample_weight=sample_weights)
        self.classes_ = self.pipe.classes_
        self.is_trained = True
        
        logger.info("Training completed")
    
    def predict(self, text: str) -> str:
        """
        Predict document type for a single text.
        
        Args:
            text: Document text
        
        Returns:
            Predicted document type
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        return self.pipe.predict([text])[0]
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict document types for multiple texts.
        
        Args:
            texts: List of document texts
        
        Returns:
            List of predicted document types
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        return self.pipe.predict(texts).tolist()
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for all classes.
        
        Args:
            text: Document text
        
        Returns:
            Dict of {class_name: probability}
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        probs = self.pipe.predict_proba([text])[0]
        return {cls: prob for cls, prob in zip(self.classes_, probs)}
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Evaluate classifier on test set.
        
        Args:
            texts: Test texts
            labels: True labels
        
        Returns:
            Dict with accuracy, classification_report, confusion_matrix
        """
        y_pred = self.predict_batch(texts)
        
        metrics = {
            "accuracy": accuracy_score(labels, y_pred),
            "classification_report": classification_report(labels, y_pred),
            "confusion_matrix": confusion_matrix(labels, y_pred),
        }
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"\n{metrics['classification_report']}")
        
        return metrics
    
    def save(self, model_path: str):
        """Save trained classifier to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipe, f)
        
        logger.info(f"Classifier saved to {model_path}")
    
    def load(self, model_path: str):
        """Load trained classifier from disk."""
        model_path = str(model_path)  # Convert Path object to string if needed
        with open(model_path, 'rb') as f:
            self.pipe = pickle.load(f)
        
        self.classes_ = self.pipe.classes_
        self.is_trained = True
        logger.info(f"Classifier loaded from {model_path}")


# =======================
# Medical Document Features
# =======================

class MedicalDocumentFeatures:
    """Extract medical document-specific features for classification."""
    
    # Keyword patterns for each document type
    KEYWORD_PATTERNS = {
        "cbc_report": {
            "keywords": ["hemoglobin", "wbc", "platelet", "hematocrit", "cbc", "blood count",
                        "complete blood count", "neutrophil", "lymphocyte", "monocyte",
                        "eosinophil", "basophil", "mcv", "mch", "mchc", "rbc count",
                        "hemoglobin g", "hb g/dl", "three part differential", "tlc",
                        "esr", "blood cell count"],
            "weight": 1.0,
        },
        "lft_report": {
            "keywords": ["bilirubin", "albumin", "sgpt", "sgot", "alkaline phosphatase", 
                        "lft", "liver function", "ast", "alt", "liver function test",
                        "total protein", "globulin", "cholesterol", "triglyceride",
                        "biochemistry", "test description", "ref. range", "hepatic",
                        "liver enzymes", "ggtp", "direct bilirubin", "indirect bilirubin"],
            "weight": 1.0,
        },
        "urine_report": {
            "keywords": ["urine", "urinalysis", "creatinine", "glucose", "protein", 
                        "rbc", "wbc", "bacteria", "urine output", "specific gravity",
                        "urinary", "urine analysis", "urine routine", "24 hour",
                        "ph", "nitrites", "leukocyte esterase"],
            "weight": 1.0,
        },
        "discharge_summary": {
            "keywords": ["discharge", "date of discharge", "clinical summary", "hospital course", 
                        "treatment", "medications", "follow-up", "diagnosis", "admitted",
                        "course in hospital", "consultant", "advice", "discharge date",
                        "date of admission", "admission", "normal discharge", "surgery",
                        "operative", "procedure", "department of"],
            "weight": 1.0,
        },
        "prescription": {
            "keywords": ["tablet", "capsule", "syrup", "injection", "dose", 
                        "frequency", "morning", "evening", "twice daily", "mg", "ml",
                        "once daily", "thrice", "ointment", "cream", "rx", "prescription date",
                        "tab", "cap", "inj", "bid", "tid", "od", "after meals", "with meals",
                        "before food", "at bedtime"],
            "weight": 1.0,
        },
        "clinical_notes": {
            "keywords": ["patient", "complaint", "examination", "clinical", 
                        "vital signs", "assessment", "plan", "impression", "session content",
                        "presenting problem", "client", "therapist", "intervention",
                        "date of session", "blood pressure", "pulse rate", "spo2",
                        "progress note", "clinical note", "subjective", "objective",
                        "soaps", "follow-up", "treatment plan", "therapy"],
            "weight": 1.0,
        },
    }
    
    @classmethod
    def extract_keyword_features(cls, text: str) -> Dict[str, float]:
        """
        Extract keyword-based features.
        
        Returns:
            Dict of {doc_type: score}
        """
        text_lower = text.lower()
        scores = {}
        
        for doc_type, config in cls.KEYWORD_PATTERNS.items():
            keywords = config["keywords"]
            weight = config["weight"]
            
            # Count keyword occurrences
            count = sum(text_lower.count(kw) for kw in keywords)
            scores[doc_type] = count * weight
        
        return scores
    
    @classmethod
    def extract_structural_features(cls, lines: List[str]) -> Dict[str, float]:
        """
        Extract structural features from document lines.
        
        Returns:
            Dict of {feature: value}
        """
        features = {
            "num_lines": len(lines),
            "avg_line_length": np.mean([len(line) for line in lines]) if lines else 0,
            "has_table_indicators": sum(1 for line in lines if "|" in line or "—" in line),
            "has_numeric_data": sum(1 for line in lines if any(c.isdigit() for c in line)),
        }
        
        return features
    
    @classmethod
    def combine_features(cls, text: str, lines: List[str]) -> Dict:
        """
        Combine all features for enhanced classification.
        
        Returns:
            Dict with keyword scores and structural features
        """
        keyword_features = cls.extract_keyword_features(text)
        structural_features = cls.extract_structural_features(lines)
        
        return {
            "keywords": keyword_features,
            "structure": structural_features,
        }


# =======================
# Ensemble Classifier
# =======================

class EnsembleDocumentClassifier:
    """
    Ensemble combining rule-based and ML-based classification.
    Rule-based gives keywords boost, ML gives final decision.
    """
    
    def __init__(self):
        self.ml_classifier = DocumentClassifier("logistic_regression", "tfidf")
        self.keyword_threshold = 0.3  # Confidence threshold for keyword-based classification
    
    def train(self, texts: List[str], labels: List[str]):
        """Train the ensemble."""
        self.ml_classifier.train(texts, labels)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict with ensemble approach: keyword-based + ML-based
        
        Returns:
            (predicted_class, confidence)
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning(f"Text too short for classification: '{text[:50]}'")
                return "unknown", 0.1
            
            # ============ Rule-based (keyword) prediction ============
            keyword_scores = MedicalDocumentFeatures.extract_keyword_features(text)
            
            # Normalize keyword scores (raw counts -> probabilities)
            total_keyword_score = sum(keyword_scores.values())
            if total_keyword_score > 0:
                keyword_proba = {k: v / total_keyword_score for k, v in keyword_scores.items()}
            else:
                keyword_proba = {k: 1.0 / len(keyword_scores) for k in keyword_scores}
            
            top_keyword_class = max(keyword_proba, key=keyword_proba.get)
            top_keyword_prob = keyword_proba[top_keyword_class]
            
            logger.info(f"Keyword-based: {top_keyword_class} ({top_keyword_prob:.2%})")
            logger.debug(f"All keyword scores: {keyword_proba}")
            
            # ============ ML-based prediction ============
            ml_proba = None
            top_ml_class = None
            top_ml_prob = 0.0
            
            if self.ml_classifier.is_trained:
                try:
                    ml_proba = self.ml_classifier.predict_proba(text)
                    if ml_proba:
                        top_ml_class = max(ml_proba, key=ml_proba.get)
                        top_ml_prob = ml_proba[top_ml_class]
                        logger.info(f"ML-based: {top_ml_class} ({top_ml_prob:.2%})")
                        logger.debug(f"All ML scores: {ml_proba}")
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
                    ml_proba = None
            else:
                logger.info("ML classifier not trained, using keyword-based only")
            
            # ============ Ensemble decision ============
            # If ML model is trained AND confident, use it
            if ml_proba and top_ml_prob >= 0.35:
                logger.info(f"Using ML (high confidence): {top_ml_class} ({top_ml_prob:.2%})")
                return top_ml_class, top_ml_prob
            # Otherwise, use keyword-based (more reliable with limited training)
            else:
                logger.info(f"Using keyword-based (ML not confident): {top_keyword_class} ({top_keyword_prob:.2%})")
                return top_keyword_class, top_keyword_prob
                
        except Exception as e:
            logger.error(f"Ensemble predict error: {e}", exc_info=True)
            # Return unknown with low confidence
            return "unknown", 0.1
    
    def save(self, model_path: str):
        """Save ensemble classifier."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.ml_classifier.save(model_path)
    
    def load(self, model_path: str):
        """Load ensemble classifier."""
        self.ml_classifier.load(model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example training
    train_texts = [
        "Patient hemoglobin 13.5 WBC 7500 platelets 250000",
        "Liver function tests bilirubin 1.0 albumin 4.0",
        "Aspirin 500mg twice daily for 5 days",
    ]
    train_labels = ["cbc_report", "lft_report", "prescription"]
    
    classifier = DocumentClassifier()
    classifier.train(train_texts, train_labels)
    
    pred = classifier.predict("WBC count 8000 hemoglobin low")
    print(f"Predicted: {pred}")
