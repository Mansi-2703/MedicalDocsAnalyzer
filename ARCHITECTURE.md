# 🏗️ Architecture Document - Medical Document Analyzer

## 📊 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     📥 INPUT LAYER                                  │
│  (Image/PDF files from healthcare providers)                        │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│         🖼️  STEP 1: PREPROCESSING LAYER (OpenCV)                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Deskew (Hough lines)                                       │  │
│  │ • Bilateral filter (noise reduction, edge preservation)      │  │
│  │ • CLAHE (contrast enhancement)                               │  │
│  │ • Otsu thresholding (binary conversion)                      │  │
│  │ • Morphological operations (dilation, erosion)               │  │
│  │ • Resize to optimal DPI (300 DPI)                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│         📝 STEP 2: OCR LAYER (Tesseract)                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Character recognition (word + line level)                  │  │
│  │ • Bounding box extraction                                    │  │
│  │ • Confidence scoring per word                                │  │
│  │ • Post-processing (medical term fixes)                       │  │
│  │ • Multi-page PDF support                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                          OUTPUT: Raw text + tokens                  │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      🏷️  STEP 3: CLASSIFICATION LAYER (Ensemble)                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Rules: Keyword→Score-based classification                   │  │
│  │  - CBC: "hemoglobin", "wbc", "platelet"                     │  │
│  │  - LFT: "bilirubin", "albumin", "sgpt", "sgot"              │  │
│  │  - Prescription: "tablet", "syrup", "dose", "frequency"     │  │
│  │                                                               │  │
│  │ ML: TF-IDF (1,2-grams) + LogisticRegression                 │  │
│  │  - Trained on document type labels                          │  │
│  │  - Feature extraction: max_features=3000                    │  │
│  │                                                               │  │
│  │ Ensemble: Rule confidence vs ML confidence                  │  │
│  │  - If rule score > threshold: use rule prediction           │  │
│  │  - Else: use ML prediction                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                   OUTPUT: document_type, confidence                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      🔍 STEP 4: NER EXTRACTION LAYER (Fine-tuned BERT)              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Model: microsoft/deberta-v3-small                            │  │
│  │  - 6 layers, 384 hidden size                                 │  │
│  │  - Efficient for small OCR datasets                          │  │
│  │                                                               │  │
│  │ BIO Tags (30 classes):                                       │  │
│  │  - Patient: PATIENT_NAME, PATIENT_ID, AGE, GENDER          │  │
│  │  - Tests: TEST_NAME, TEST_VALUE, UNIT, REF_RANGE, FLAG     │  │
│  │  - Meds: MEDICATION_NAME, DOSAGE, FREQ, DURATION, ROUTE    │  │
│  │  - Clinical: DIAGNOSIS, IMPRESSION, ABNORMAL_VALUE          │  │
│  │                                                               │  │
│  │ Training:                                                    │  │
│  │  - Hugging Face Trainer with validation loss               │  │
│  │  - Warmup 10%, LR 5e-5, epochs 3, batch 8                  │  │
│  │  - Weight decay 0.01                                        │  │
│  │                                                               │  │
│  │ Inference:                                                   │  │
│  │  - Tokenize input, predict per token                        │  │
│  │  - Confidence scoring (softmax probabilities)                │  │
│  │  - Entity alignment (B-I tagging)                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│              OUTPUT: entities_by_type = {TAG: [values]}             │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      🗂️  STEP 5: SCHEMA MAPPING LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Document Type → Schema Mapping (config.SCHEMA_MAPPING)       │  │
│  │                                                               │  │
│  │ CBC Report:                                                  │  │
│  │   patient: {name, id, age, gender}                          │  │
│  │   hospital: {name, department, doctor}                      │  │
│  │   tests: [{name, value, unit, reference, flag}]            │  │
│  │   impression: string                                        │  │
│  │                                                               │  │
│  │ Discharge Summary:                                           │  │
│  │   patient: {name, id, age, gender}                          │  │
│  │   hospital: {name, department}                              │  │
│  │   diagnosis: [list]                                         │  │
│  │   medications: [{name, dosage, frequency, duration}]        │  │
│  │   impression: string                                        │  │
│  │                                                               │  │
│  │ Prescription:                                                │  │
│  │   patient: {name, id}                                       │  │
│  │   doctor: {name}                                            │  │
│  │   medications: [{name, dosage, freq, route, duration}]      │  │
│  │                                                               │  │
│  │ Value Normalization:                                         │  │
│  │   - Gender: M/F → Male/Female                               │  │
│  │   - Age: extract numeric value                              │  │
│  │   - Units: normalize to standard (g/dL, /μL, etc)          │  │
│  │   - Numbers: extract from "13.5 g/dL" → 13.5              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          OUTPUT: structured_output = {field: value}                │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      ✅ STEP 6: VALIDATION & RULE ENGINE LAYER                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Required Fields Check:                                       │  │
│  │   - cbc_report: patient_name, patient_id, tests             │  │
│  │   - prescription: patient_name, medications                  │  │
│  │                                                               │  │
│  │ Lab Value Validation (config.VALIDATION_RULES):             │  │
│  │   - Normal range checks (gender-aware for Hgb, RBC)         │  │
│  │   - Critical value detection                                │  │
│  │   - Unit validation                                         │  │
│  │   Example: Hemoglobin                                       │  │
│  │     Male normal: 13.0-16.0 g/dL                           │  │
│  │     Female normal: 12.0-15.5 g/dL                          │  │
│  │     Critical low: 7.0, Critical high: 20.0                 │  │
│  │                                                               │  │
│  │ Data Quality Checks:                                         │  │
│  │   - Patient ID format validation                             │  │
│  │   - Age bounds (0-150 years)                                │  │
│  │   - Consistency rules (Hct ≈ 3 × Hemoglobin)               │  │
│  │   - Missing field warnings                                  │  │
│  │                                                               │  │
│  │ Output:                                                      │  │
│  │   abnormalities: [{test, value, type, severity, message}]   │  │
│  │   quality_warnings: [{field, issue, confidence, suggestion}]│  │
│  └──────────────────────────────────────────────────────────────┘  │
│        OUTPUT: abnormalities[], quality_warnings[], is_valid        │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      🤖 STEP 7: LLM SUMMARIZATION LAYER (Optional)                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ LLM Providers (fallback chain):                              │  │
│  │   1. OpenAI GPT-4 API (if OPENAI_API_KEY set)               │  │
│  │   2. Hugging Face local LLM (if available)                  │  │
│  │   3. Rule-based fallback summarizer                         │  │
│  │                                                               │  │
│  │ Prompt Templates (structured data ONLY):                    │  │
│  │   - For CBC: "Generate 2-3 sentence clinical summary"      │  │
│  │   - For LFT: "Highlight liver function pattern"             │  │
│  │   - For Discharge: "3-4 sentence summary with diagnosis"    │  │
│  │   - For Rx: "Medication summary with key instructions"     │  │
│  │                                                               │  │
│  │ Fallback (if no LLM):                                        │  │
│  │   - Count abnormalities, list test names                    │  │
│  │   - Count medications, list names                           │  │
│  │   - Template-based sentence generation                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                OUTPUT: clinical_summary = string                    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│      🌐 STEP 8: API RESPONSE LAYER (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ InferenceResponse (Pydantic model):                          │  │
│  │   request_id: str                                           │  │
│  │   timestamp: str (ISO format)                               │  │
│  │   document_type: str + confidence: float                    │  │
│  │   patient: PatientInfo                                      │  │
│  │   structured_output: Dict (document-type specific)          │  │
│  │   validation_results: Dict                                  │  │
│  │   abnormalities: [AbnormalityInfo]                         │  │
│  │   quality_warnings: [QualityIssue]                         │  │
│  │   clinical_summary: str                                     │  │
│  │   processing_time_ms: float                                │  │
│  │   ocr_confidence: float (optional)                         │  │
│  │   ocr_text: str (optional)                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│ 📡 API ENDPOINTS:                                                   │
│   📤 POST /analyze          - Full pipeline (8 steps)              │
│   🏷️  POST /classify         - Quick classification only            │
│   💚 GET  /health           - Health check + component status      │
│   📊 GET  /models/status    - Model loading status                 │
└────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  JSON RESPONSE   │
            │  (Client)        │
            └──────────────────┘
```

---

## 🔄 Data Flow & Dependencies

```
healthcare_dataset.json
│
├─→ [Training Pipeline]
│   ├─→ DataProcessor.split_dataset()
│   │   └─→ train (70%), val (15%), test (15%)
│   │
│   ├─→ ClassifierTrainer.train()
│   │   ├─→ TF-IDF vectorizer
│   │   ├─→ LogisticRegression classifier
│   │   └─→ models/classifier/classifier.pkl
│   │
│   └─→ NERTrainer.train()
│       ├─→ DeBERTa-v3-small tokenizer
│       ├─→ Hugging Face Trainer
│       └─→ models/ner_model/
│
└─→ [Inference Pipeline]
    ├─→ ImagePreprocessor.preprocess_pipeline()
    │   └─→ preprocessed_image
    │
    ├─→ OCREngine.extract_structured()
    │   └─→ raw_text, tokens, words, confidence
    │
    ├─→ EnsembleDocumentClassifier.predict()
    │   ├─→ MedicalDocumentFeatures.extract_keyword_features()
    │   ├─→ DocumentClassifier.predict_proba()
    │   └─→ doc_type, confidence
    │
    ├─→ MedicalNERModel.predict()
    │   ├─→ Tokenize tokens
    │   ├─→ DeBERTa forward pass
    │   └─→ predicted_tags
    │
    ├─→ extract_entities()
    │   └─→ entities_by_type
    │
    ├─→ SchemaMapper.map_to_structured()
    │   ├─→ ValueNormalizer methods
    │   └─→ structured_output
    │
    ├─→ RuleEngine.validate()
    │   └─→ abnormalities, quality_warnings
    │
    ├─→ MedicalDocumentSummarizer.summarize()
    │   └─→ clinical_summary
    │
    └─→ InferenceResponse
```

---

## 🔗 Component Interactions

### During Training:
```
healthcare_dataset.json
  ↓
DataProcessor
  ├─ Load JSON
  ├─ Split train/val/test
  ├─ Extract texts & labels → ClassifierTrainer
  └─ Extract tokens & tags → NERTrainer
```

### During Inference:
```
Image/PDF
  ↓
ImagePreprocessor (OpenCV)
  ↓
OCREngine (Tesseract)
  ↓
EnsembleDocumentClassifier (TF-IDF + LogReg)
  ↓
MedicalNERModel (DeBERTa)
  ↓
extract_entities()
  ↓
SchemaMapper
  ↓
RuleEngine
  ├─→ abnormalities
  ├─→ quality_warnings
  └─→ validation_results
  ↓
MedicalDocumentSummarizer (OpenAI/HF/Fallback)
  ↓
FastAPI Response
```

---

## ⚙️ Model Specifications

### 📊 Document Classifier
- **Algorithm:** TF-IDF Vectorizer + Logistic Regression
- **Input:** Raw OCR text
- **Output:** document_type, confidence
- **Parameters:**
  - vectorizer: max_features=3000, ngram_range=(1,2)
  - classifier: max_iter=200, multi_class='multinomial'
- **Training:** texts & labels from dataset
- **Size:** ~5MB pickle file

### 🧠 NER Model
- **Architecture:** DeBERTa-v3-small (decoder-enhanced BERT)
- **Input:** Token sequences (max_len=512)
- **Output:** BIO tags + confidence
- **Parameters:**
  - num_labels: 30 (BIO entity types)
  - learning_rate: 5e-5
  - num_epochs: 3
  - warmup_ratio: 0.1
  - batch_size: 8
- **Training:** tokens & BIO tags from dataset
- **Size:** ~270MB (model weights + tokenizer)

### ✔️ Rule Engine
- **Type:** Rule-based validation (no ML)
- **Input:** structured_output + patient_demographics
- **Output:** abnormalities, quality_warnings
- **Rules:** Hardcoded medical thresholds from VALIDATION_RULES
- **Performance:** O(n) where n=number of tests

---

## 🚨 Error Handling & Fallbacks

```
Preprocessing Error
  └─→ Log & raise for user - image too corrupted

OCR Failure (no text extracted)
  └─→ Log warning, proceed with empty text field

Classification Error
  └─→ Default to "unknown" type with 0% confidence

NER Error (model not loaded)
  └─→ Log warning, return empty entities

Schema Mapping Error
  └─→ Return generic schema with extracted fields

Validation Error
  └─→ Log warning, continue - non-critical

LLM Summarization Error
  └─→ Use FallbackSummarizer or skip summary

API Error
  └─→ Return 500 with error message
```

---

## 🔋 Computing Resource Requirements

### 📋 Training
- **RAM:** 4-8 GB minimum
- **GPU:** Optional (NVIDIA CUDA 11.8+)
  - Without GPU: ~15 min per epoch on 3000 samples
  - With GPU: ~30 sec per epoch
- **Storage:** 20 GB for models + dataset

### ⚡ Inference
- **RAM:** 2-4 GB
- **GPU:** Optional (reduces latency 3-5x)
- **Storage:** 2 GB for model files
- **Latency:** 1-3 sec per document on CPU, 0.3-0.8 sec on GPU

---

## 🧹 Extensibility

### 📁 Add New Document Type
1. Add to `DOCUMENT_TYPES` in config.py
2. Define schema in `SCHEMA_MAPPING`
3. Add validation rules in `VALIDATION_RULES`
4. Add LLM prompt template in summarizer.py
5. Retrain with documents of new type

### 🌟 Add New NER Entity Type
1. Add B-TAG and I-TAG to `NER_TAGS` in config.py
2. Add training samples with new tags
3. Retrain NER model
4. Update schema mapping if needed

### 📚 Support New LLM Provider
1. Create class inheriting `BaseLLMClient`
2. Implement `generate_summary()` method
3. Add to provider selection in `MedicalDocumentSummarizer.__init__`

---

## ⚡️ Performance Optimization Tips

1. **Preprocessing:** Use `aggressive=False` unless OCR quality is very poor
2. **NER Batch:** Process multiple documents → batch inference (faster)
3. **Caching:** Cache classifier vectorizer and NER model in memory
4. **GPU:** Use CUDA for NER inference if available
5. **API:** Enable uvicorn workers for multi-document parallelism

---

## 🔐 Security Considerations

1. **API:** Validate file types (only .jpg, .png, .pdf)
2. **File Storage:** Cleanup temp files after processing
3. **Data Privacy:** Don't store raw OCR text in logs
4. **LLM API Key:** Store in environment variables only
5. **Model Files:** Validate checksum before loading

