"""
Configuration for Medical Document Analyzer pipeline.
Optimized for small OCR-heavy mixed-schema medical datasets.
"""

import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "training"

# =======================
# Data Paths
# =======================
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
ANNOTATIONS_PATH = DATA_DIR / "annotations"

# =======================
# Model Paths
# =======================
NER_MODEL_PATH = MODELS_DIR / "ner_model"  # Directory path (not specific file)
NER_CONFIG_PATH = MODELS_DIR / "ner_model" / "config.json"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier" / "classifier.pkl"

# =======================
# OCR & Preprocessing
# =======================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# TESSERACT_PATH = "/usr/bin/tesseract"  # Linux/Mac

PREPROCESSING_CONFIG = {
    "image_dpi": 300,
    "threshold_value": 127,
    "morph_kernel_size": (5, 5),
    "dilation_iterations": 2,
    "bilateral_filter_enabled": True,
}

# =======================
# Document Types & Tags
# =======================
DOCUMENT_TYPES = {
    "cbc_report": {"display": "CBC Report", "priority": 1},
    "lft_report": {"display": "Liver Function Test", "priority": 1},
    "urine_report": {"display": "Urine Report", "priority": 1},
    "mixed_diagnostic": {"display": "Mixed Diagnostic", "priority": 2},
    "discharge_summary": {"display": "Discharge Summary", "priority": 2},
    "prescription": {"display": "Prescription", "priority": 3},
    "clinical_notes": {"display": "Clinical Notes", "priority": 3},
}

SOURCE_STYLES = [
    "discharge_summary_ocr",
    "lab_report_scanned",
    "prescription_typed",
    "clinical_notes_handwritten",
    "mixed_format",
]

# =======================
# NER Tags (BIO Format)
# =======================
NER_TAGS = [
    "O",  # Outside
    "B-PATIENT_NAME",
    "I-PATIENT_NAME",
    "B-PATIENT_ID",
    "B-AGE",
    "B-GENDER",
    "B-HOSPITAL_NAME",
    "I-HOSPITAL_NAME",
    "B-REPORT_DATE",
    "B-TEST_NAME",
    "I-TEST_NAME",
    "B-TEST_VALUE",
    "B-UNIT",
    "B-REFERENCE_RANGE",
    "B-FLAG",
    "B-ABNORMAL_VALUE",
    "B-CLINICAL_IMPRESSION",
    "I-CLINICAL_IMPRESSION",
    "B-DIAGNOSIS",
    "I-DIAGNOSIS",
    "B-MEDICATION_NAME",
    "I-MEDICATION_NAME",
    "B-DOSAGE",
    "B-FREQUENCY",
    "B-DURATION",
    "B-ROUTE",
    "B-INSTRUCTION",
    "I-INSTRUCTION",
    "B-DEPARTMENT",
    "B-DOCTOR_NAME",
]

TAG_TO_ID = {tag: idx for idx, tag in enumerate(NER_TAGS)}
ID_TO_TAG = {idx: tag for idx, tag in enumerate(NER_TAGS)}

# =======================
# NER Model Config
# =======================
NER_MODEL_CONFIG = {
    "model_name": "microsoft/deberta-v3-small",
    "max_seq_length": 512,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "seed": 42,
    "save_strategy": "epoch",
}

# =======================
# Classification Config
# =======================
CLASSIFIER_CONFIG = {
    "model_type": "logistic_regression",  # or 'random_forest', 'svm'
    "vectorizer": "tfidf",  # tfidf or count
    "max_features": 3000,
    "ngram_range": (1, 2),
}

# =======================
# Schema Mapping Rules
# =======================
# Maps extracted NER tags to structured output fields
SCHEMA_MAPPING = {
    "cbc_report": {
        "patient_demographics": {
            "patient_name": "B-PATIENT_NAME",
            "patient_id": "B-PATIENT_ID",
            "age": "B-AGE",
            "gender": "B-GENDER",
        },
        "hospital_info": {
            "hospital_name": "B-HOSPITAL_NAME",
            "department": "B-DEPARTMENT",
        },
        "test_info": {
            "report_date": "B-REPORT_DATE",
            "tests": {
                "test_name": "B-TEST_NAME",
                "value": "B-TEST_VALUE",
                "unit": "B-UNIT",
                "reference_range": "B-REFERENCE_RANGE",
                "flag": "B-FLAG",
            },
        },
        "impression": {
            "clinical_impression": "B-CLINICAL_IMPRESSION",
        },
    },
    "discharge_summary": {
        "patient_demographics": {
            "patient_name": "B-PATIENT_NAME",
            "patient_id": "B-PATIENT_ID",
            "age": "B-AGE",
            "gender": "B-GENDER",
        },
        "hospital_info": {
            "hospital_name": "B-HOSPITAL_NAME",
            "department": "B-DEPARTMENT",
        },
        "clinical_info": {
            "diagnosis": "B-DIAGNOSIS",
            "treatments": "B-MEDICATION_NAME",
            "clinical_impression": "B-CLINICAL_IMPRESSION",
        },
    },
    "prescription": {
        "patient_demographics": {
            "patient_name": "B-PATIENT_NAME",
            "patient_id": "B-PATIENT_ID",
        },
        "hospital_info": {
            "doctor_name": "B-DOCTOR_NAME",
            "hospital_name": "B-HOSPITAL_NAME",
        },
        "medications": {
            "medication_name": "B-MEDICATION_NAME",
            "dosage": "B-DOSAGE",
            "frequency": "B-FREQUENCY",
            "duration": "B-DURATION",
            "route": "B-ROUTE",
            "instruction": "B-INSTRUCTION",
        },
    },
}

# =======================
# Validation Rules (Rule Engine)
# =======================
VALIDATION_RULES = {
    "cbc_report": {
        "hemoglobin": {
            "normal_range_male": (13.0, 16.0),
            "normal_range_female": (12.0, 15.5),
            "unit": "g/dL",
            "critical_low": 7.0,
            "critical_high": 20.0,
        },
        "hematocrit": {
            "normal_range_male": (41, 53),
            "normal_range_female": (36, 46),
            "unit": "%",
            "critical_low": 20,
            "critical_high": 60,
        },
        "mcv": {
            "normal_range": (76, 96),
            "unit": "fL",
        },
        "mch": {
            "normal_range": (26, 32),
            "unit": "pg",
        },
        "mchc": {
            "normal_range": (32, 36),
            "unit": "g/dL",
        },
        "wbc": {
            "normal_range": (4500, 11000),
            "unit": "/μL",
            "critical_low": 2000,
            "critical_high": 30000,
        },
        "platelets": {
            "normal_range": (150000, 400000),
            "unit": "/μL",
            "critical_low": 50000,
            "critical_high": 1000000,
        },
        "rbc": {
            "normal_range_male": (4.5, 5.9),
            "normal_range_female": (4.1, 5.1),
            "unit": "million/μL",
            "critical_low": 2.0,
            "critical_high": 8.0,
        },
        "esr": {
            "normal_range_male": (0, 20),
            "normal_range_female": (0, 30),
            "unit": "mm/hr",
        },
        "neutrophil": {
            "normal_range": (40, 75),
            "unit": "%",
        },
        "lymphocyte": {
            "normal_range": (20, 50),
            "unit": "%",
        },
        "monocyte": {
            "normal_range": (2, 10),
            "unit": "%",
        },
        "eosinophil": {
            "normal_range": (0, 5),
            "unit": "%",
        },
        "basophil": {
            "normal_range": (0, 1),
            "unit": "%",
        },
    },
    "lft_report": {
        "bilirubin_total": {
            "normal_range": (0.1, 1.2),
            "unit": "mg/dL",
            "critical_high": 10.0,
        },
        "albumin": {
            "normal_range": (3.5, 5.5),
            "unit": "g/dL",
            "critical_low": 2.0,
        },
        "sgpt": {
            "normal_range": (0, 41),
            "unit": "U/L",
            "critical_high": 1000,
        },
        "sgot": {
            "normal_range": (0, 40),
            "unit": "U/L",
            "critical_high": 1000,
        },
    },
    "prescription": {
        "must_have_fields": ["medication_name", "dosage", "frequency"],
        "optional_fields": ["instruction", "duration", "route"],
    },
}

# =======================
# LLM Config
# =======================
LLM_CONFIG = {
    "provider": "openai",  # openai, azure, huggingface
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 500,
    "api_key": os.getenv("OPENAI_API_KEY", ""),
}

# =======================
# Logging
# =======================
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }
    },
}
