"""
Init file for classification module.
"""

from .doc_classifier import DocumentClassifier, MedicalDocumentFeatures, EnsembleDocumentClassifier

__all__ = [
    "DocumentClassifier",
    "MedicalDocumentFeatures",
    "EnsembleDocumentClassifier",
]
