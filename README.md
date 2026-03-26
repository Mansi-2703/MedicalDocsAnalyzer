# 🏥 Medical Document Analyzer

> **Intelligent OCR + AI-Powered Medical Document Processing Pipeline**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)](#status)
[![Accuracy](https://img.shields.io/badge/Classification%20Accuracy-94.25%25-orange.svg)](#performance)

---

## 📋 Overview

Medical Document Analyzer is an **end-to-end intelligent document processing system** that transforms medical reports into structured, patient-friendly clinical insights using advanced OCR, AI/ML, and LLM technologies.

### ⚡ What It Does

```
📄 Image  →  🖼️ Preprocess  →  🔤 OCR  →  🏷️ Classify  →  🔍 Extract  
→  📊 Map  →  ⚠️ Detect  →  💬 Summarize  →  📦 JSON
```

---

## ✨ Key Highlights

- 📚 **13+ Document Types**: CBC, LFT, KFT, Prescriptions, Discharge, etc.
- 🎯 **94.25% Classification Accuracy** with balanced class weights
- 🔍 **Entity Extraction**: Fine-tuned BERT for 30+ medical entities
- ⚠️ **Abnormality Detection**: 50+ lab tests with severity flagging (CRITICAL/HIGH/LOW/NORMAL)
- 💭 **AI Summaries**: Google Gemini API + rule-based fallback
- ⚡ **Fast & Scalable**: <2s per document, comprehensive error handling, production-ready
- 🔒 **Patient-Friendly**: Age/gender-aware reference ranges with plain-language explanations

---

##  Quick Start

### 📋 Prerequisites
```bash
Python 3.9+
Tesseract OCR (system dependency)
CUDA 12.4 (optional, for GPU)
```

### ⚙️ Installation

```bash
# 1. Clone & Setup Environment
cd MedicalDocAnalyzer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Configure API Keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Prepare Models
# NER & classifier models are in .gitignore (too large for GitHub)
# Models are auto-loaded on first run if available
# To train/retrain: python training/retrain_classifier.py

# 5. Start Server
python -m uvicorn src.api.fastapi_app:app --reload --port 8000
```

### 🌐 Access the API
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

### 📦 About Models
Models are excluded from git (415 MB each) but auto-loaded on first run:
- **Classifier**: `models/classifier/` (trained on 87 medical documents)
- **NER Model**: `models/ner_model/` (fine-tuned BERT)

To retrain the classifier with your own data:
```bash
python training/retrain_classifier.py
```

---

## 📡 API Endpoints

### 🏷️ Quick Classification
```bash
POST /classify
Content-Type: multipart/form-data

Response: {
  "document_type": "cbc_report",
  "confidence": 0.97,
  "timestamp": "2026-03-26T01:13:33"
}
```

### 📊 Full Document Analysis
```bash
POST /analyze
Parameters:
  - file: medical document image
  - include_llm_summary: true/false
  - include_validation: true/false
  - return_raw_ocr: true/false

Response: {
  "request_id": "550e8400-e29b",
  "document_type": "lft_report",
  "patient": { "name": "John", "age": "45" },
  "structured_output": {
    "tests": [{
      "test_name": "bilirubin_total",
      "value": "1.8",
      "severity": "HIGH",
      "patient_explanation": "Your bilirubin..."
    }]
  },
  "abnormality_flags": [...],
  "llm_summary": "Patient analysis..."
}
```

### 🔧 Model Status
```bash
GET /models/status

{
  "classifier": { "status": "loaded", "accuracy": 0.9425 },
  "ner_model": { "status": "loaded", "type": "BERT" },
  "llm_client": { "status": "initialized", "model": "gemini-2.0" }
}
```

---

## 🏗️ Architecture

```
📄 Input Image  
  ↓ Preprocess & OCR (Tesseract)
  ↓ Classify Document (94.25% accuracy)
  ↓ Extract Entities (BERT NER)
  ↓ Detect Abnormalities (50+ tests)
  ↓ Summarize & Validate (Gemini + rules)
  ↓ NDJSON Output
```

**Core Modules**: 
- `src/preprocessing/` - Image & OCR processing
- `src/classification/` - Document type detection
- `src/extraction/` - NER entity extraction
- `src/mapping/` - Abnormality detection & schema mapping
- `src/validation/` - Clinical rule validation
- `src/llm/` - LLM-based summarization
- `src/api/` - FastAPI endpoints

---

## 📚 Project Structure

```
MedicalDocAnalyzer/
├── README.md, ARCHITECTURE.md      Documentation
├── requirements.txt, config.py     Dependencies & config
├── .env.example, .gitignore        Setup templates
│
├── src/                            Production code
│   ├── api/                 FastAPI endpoints
│   ├── preprocessing/       Image & OCR
│   ├── classification/      Document detection
│   ├── extraction/          NER entity extraction
│   ├── mapping/             Abnormality detection
│   ├── validation/          Rule engine
│   └── llm/                 LLM summarization
│
├── models/                         Pre-trained models
│   ├── classifier/          (Auto-downloaded on first run)
│   └── ner_model/           (Auto-downloaded on first run)
│
├── training/                       Training utilities
│   └── retrain_classifier.py       Retraining script
│
└── data/                           Datasets
    ├── raw/                 Raw medical documents
    └── processed/           Preprocessed images
```

---

## 🛠️ Usage Examples

### Python API
```python
from src.classification import EnsembleDocumentClassifier
from src.mapping import AbnormalityDetector

# Quick classification
classifier = EnsembleDocumentClassifier()
doc_type, confidence = classifier.predict("CBC report text...")
print(f"Document: {doc_type} ({confidence:.2%})")

# Abnormality detection
detector = AbnormalityDetector()
abnormality = detector.detect_abnormality(
    test_name="hemoglobin",
    value=7.5,
    gender="Female",
    age=35
)
print(abnormality.patient_explanation)
```

### REST API
```bash
# Classify document
curl -X POST "http://localhost:8000/classify" \
  -F "file=@medical_report.jpg"

# Full analysis
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@medical_report.jpg" \
  -F "include_llm_summary=true"

# Check models
curl "http://localhost:8000/models/status"
```

---

## 🎓 Supported Document Types

Lab Reports: CBC, LFT, KFT, Blood Sugar, Lipid Profile, Thyroid, Urine Analysis  
Clinical: Discharge Summaries, Clinical Notes, Diagnostic Reports  
Medication: Prescriptions, Medication Lists

---

## 📖 Documentation

- 📘 [Full Architecture](ARCHITECTURE.md)
- 🚀 [API Docs](http://localhost:8000/docs) (when server running)

---

<div align="center">

**Medical Document Analyzer v2.0** — Production-ready OCR + AI for healthcare  
Built with ❤️ for better medical document processing

⭐ ***Star if you find it useful!***

</div>
