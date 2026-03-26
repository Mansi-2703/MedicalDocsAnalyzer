"""
FastAPI Endpoint for Medical Document Analysis Pipeline.
Provides REST API for complete document processing.
"""

# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# ===========================
# Request/Response Models
# ===========================

class InferenceRequest(BaseModel):
    """Request model for document analysis."""
    document_type: Optional[str] = None  # Auto-detect if not provided
    include_llm_summary: bool = True
    include_validation: bool = True
    return_raw_ocr: bool = False


class TestResult(BaseModel):
    """Single test result in response."""
    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None


class PatientInfo(BaseModel):
    """Patient information in response."""
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None


class AbnormalityInfo(BaseModel):
    """Abnormality/Alert information."""
    test_name: str
    value: Any
    type: str
    severity: str
    message: str


class QualityIssue(BaseModel):
    """Data quality issue."""
    field: str
    issue: str
    confidence: float
    suggestion: str


class InferenceResponse(BaseModel):
    """Response model for document analysis."""
    request_id: str
    timestamp: str
    document_type: str
    document_type_confidence: float
    
    # Patient info
    patient: PatientInfo
    
    # Structured output (document-type specific)
    structured_output: Dict[str, Any]
    
    # Validation results
    validation_results: Optional[Dict[str, Any]] = None
    abnormalities: Optional[List[AbnormalityInfo]] = None
    quality_warnings: Optional[List[QualityIssue]] = None
    
    # LLM summary
    clinical_summary: Optional[str] = None
    
    # Optional raw OCR data
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    
    # Metadata
    processing_time_ms: float
    warnings: List[str] = []


# ===========================
# API Setup
# ===========================

def create_app(config_path: str = None) -> FastAPI:
    """
    Create and configure FastAPI app.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configured FastAPI application
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = FastAPI(
        title="Medical Document Analyzer API",
        description="OCR + Classification + NER + Validation + LLM Summary",
        version="1.0.0",
    )
    
    # Initialize components
    from src.preprocessing import ImagePreprocessor, OCREngine
    from src.classification import EnsembleDocumentClassifier
    from src.extraction import MedicalNERModel
    from src.mapping import SchemaMapper
    from src.validation import RuleEngine
    from src.llm import MedicalDocumentSummarizer, FallbackSummarizer
    from config import (
        PREPROCESSING_CONFIG, NER_MODEL_PATH, CLASSIFIER_MODEL_PATH,
        TAG_TO_ID, ID_TO_TAG, SCHEMA_MAPPING, VALIDATION_RULES,
        TESSERACT_PATH
    )
    import uuid
    import time
    
    # Create instances
    preprocessor = ImagePreprocessor(PREPROCESSING_CONFIG)
    ocr_engine = OCREngine(tesseract_path=TESSERACT_PATH)
    classifier = EnsembleDocumentClassifier()
    ner_model = MedicalNERModel(tag_to_id=TAG_TO_ID, id_to_tag=ID_TO_TAG)
    schema_mapper = SchemaMapper()
    rule_engine = RuleEngine(VALIDATION_RULES)
    
    # Try to load pre-trained models
    try:
        classifier.load(CLASSIFIER_MODEL_PATH)
        logger.info("Classifier model loaded")
    except Exception as e:
        logger.warning(f"Classifier model not found: {e} - will use keyword-based classification")
    
    try:
        ner_model.load(NER_MODEL_PATH)
        logger.info("NER model loaded")
    except Exception as e:
        logger.warning(f"NER model not found: {e} - will need training")
    
    # LLM summarizer with fallback
    # Try Gemini first (free tier available), then fallback to other providers or rule-based
    from src.llm.summarizer import LLMProvider
    
    llm_summarizer = None
    llm_provider_used = "fallback"
    
    # Try Gemini (recommended - free tier)
    if os.getenv("GEMINI_API_KEY"):
        try:
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            from src.llm.summarizer import GeminiClient
            gemini_client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"), model=gemini_model)
            llm_summarizer = MedicalDocumentSummarizer(llm_client=gemini_client)
            llm_provider_used = "gemini"
            logger.info(f"Gemini LLM client initialized (model: {gemini_model})")
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e} - will try OpenAI")
    
    # Try OpenAI if Gemini not available
    if not llm_summarizer and os.getenv("OPENAI_API_KEY"):
        try:
            from src.llm.summarizer import OpenAIClient
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=openai_model)
            llm_summarizer = MedicalDocumentSummarizer(llm_client=openai_client)
            llm_provider_used = "openai"
            logger.info(f"OpenAI LLM client initialized (model: {openai_model})")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")
    
    # If no LLM provider available, use fallback
    if not llm_summarizer:
        logger.warning("No LLM API key configured (GEMINI_API_KEY or OPENAI_API_KEY) - will use rule-based fallback summarizer")
        logger.info("To enable AI summaries:")
        logger.info("  1. Get free Gemini API key at: https://ai.google.dev/")
        logger.info("  2. Add to .env: GEMINI_API_KEY=your_key")
        logger.info("  3. Restart the server")
    
    fallback_summarizer = FallbackSummarizer()
    
    # ===========================
    # Endpoints
    # ===========================
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize on startup."""
        logger.info("Medical Document Analyzer API started")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "preprocessor": "ok",
                "ocr": "ok",
                "classifier": "loaded" if classifier.ml_classifier.is_trained else "not_trained",
                "ner": "loaded" if ner_model.is_trained else "not_trained",
                "llm": f"available ({llm_provider_used})" if llm_summarizer else f"fallback ({llm_provider_used})",
            }
        }
    
    @app.post("/analyze", response_model=InferenceResponse)
    async def analyze_document(
        file: UploadFile = File(...),
        document_type: Optional[str] = None,
        include_llm_summary: bool = True,
        include_validation: bool = True,
        return_raw_ocr: bool = True,
        background_tasks: BackgroundTasks = BackgroundTasks()
    ):
        """
        Analyze medical document end-to-end.
        
        Process:
        1. Save uploaded file
        2. Preprocess image
        3. OCR extraction
        4. Document classification
        5. NER entity extraction
        6. Schema mapping
        7. Validation & rule engine
        8. LLM summary generation
        """
        # Create request object from parameters
        request = InferenceRequest(
            document_type=document_type,
            include_llm_summary=include_llm_summary,
            include_validation=include_validation,
            return_raw_ocr=return_raw_ocr
        )
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"[{request_id}] Analyzing document: {file.filename}")
        
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Step 1: Preprocessing
            logger.info(f"[{request_id}] Step 1: Preprocessing")
            processed_image = preprocessor.preprocess_pipeline(tmp_path)
            
            # Step 2: OCR
            logger.info(f"[{request_id}] Step 2: OCR extraction")
            ocr_result = ocr_engine.extract_structured(processed_image)
            raw_text = ocr_result.raw_text
            ocr_confidence = ocr_result.confidence
            
            # Step 3: Classification
            logger.info(f"[{request_id}] Step 3: Document classification")
            detected_doc_type, doc_type_confidence = classifier.predict(raw_text)
            doc_type = request.document_type or detected_doc_type
            
            logger.info(f"[{request_id}] Detected type: {doc_type} ({doc_type_confidence:.2%})")
            
            # Step 4: NER Extraction
            logger.info(f"[{request_id}] Step 4: NER entity extraction")
            tokens = ocr_result.words  # Use tokens from OCR
            token_texts = [t["text"] for t in tokens] if isinstance(tokens[0], dict) else tokens
            
            predicted_tags = ner_model.predict(token_texts)
            
            # Extract entities
            from src.extraction import extract_entities
            entities_by_type = extract_entities(token_texts, predicted_tags)
            
            # Step 5: Schema Mapping
            logger.info(f"[{request_id}] Step 5: Schema mapping")
            structured_output = schema_mapper.map_to_structured(doc_type, entities_by_type, raw_text)
            
            # Step 6: Validation
            validation_results = None
            abnormalities = []  # Default to empty list
            quality_warnings = []  # Default to empty list
            
            if request.include_validation:
                logger.info(f"[{request_id}] Step 6: Validation & Rule Engine")
                validation_results = rule_engine.validate(
                    doc_type,
                    structured_output,
                    {"gender": structured_output.get("gender")}
                )
                
                # Extract abnormalities and quality warnings from validation results
                if validation_results:
                    abnormalities = [AbnormalityInfo(**abn) for abn in validation_results.get("abnormalities", [])]
                    quality_warnings = [QualityIssue(**warn) for warn in validation_results.get("quality_warnings", [])]
                    
                    logger.info(f"[{request_id}] Validation found {len(abnormalities)} abnormalities, {len(quality_warnings)} quality warnings")
            else:
                # If validation not requested, return None instead of empty lists
                abnormalities = None
                quality_warnings = None
            
            # Step 7: LLM Summary
            clinical_summary = None
            if request.include_llm_summary:
                logger.info(f"[{request_id}] Step 7: LLM summary generation")
                try:
                    if llm_summarizer:
                        clinical_summary = llm_summarizer.summarize(doc_type, structured_output)
                    else:
                        # Use fallback summarizer for any document type
                        clinical_summary = fallback_summarizer.summarize(doc_type, structured_output)
                except Exception as e:
                    logger.warning(f"[{request_id}] LLM summary generation failed: {e}")
                    # Fall back to rule-based summarizer if LLM fails
                    try:
                        clinical_summary = fallback_summarizer.summarize(doc_type, structured_output)
                    except Exception as fallback_e:
                        logger.error(f"[{request_id}] Fallback summarizer also failed: {fallback_e}")
                        clinical_summary = None
            
            # Build response
            processing_time_ms = (time.time() - start_time) * 1000
            
            response = InferenceResponse(
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                document_type=doc_type,
                document_type_confidence=doc_type_confidence,
                patient=PatientInfo(
                    patient_name=structured_output.get("patient_name"),
                    patient_id=structured_output.get("patient_id"),
                    age=structured_output.get("age"),
                    gender=structured_output.get("gender"),
                ),
                structured_output=structured_output,
                validation_results=validation_results,
                abnormalities=abnormalities,
                quality_warnings=quality_warnings,
                clinical_summary=clinical_summary,
                ocr_text=raw_text if request.return_raw_ocr else None,
                ocr_confidence=ocr_confidence if request.return_raw_ocr else None,
                processing_time_ms=processing_time_ms,
                warnings=[f"Processing completed in {processing_time_ms:.0f}ms"],
            )
            
            logger.info(f"[{request_id}] Analysis completed in {processing_time_ms:.0f}ms")
            
            # Clean up
            background_tasks.add_task(Path(tmp_path).unlink)
            
            return response
        
        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    @app.post("/classify")
    async def classify_only(file: UploadFile = File(...)):
        """Quick document classification endpoint."""
        tmp_path = None
        try:
            content = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Quick OCR + Classification
            logger.info(f"Classifying document: {file.filename}")
            
            try:
                processed = preprocessor.preprocess_pipeline(tmp_path)
                logger.info("Image preprocessing completed")
            except Exception as e:
                logger.error(f"Preprocessing error: {e}", exc_info=True)
                raise
            
            try:
                ocr_result = ocr_engine.extract_structured(processed)
                logger.info(f"OCR extraction completed. Raw text length: {len(ocr_result.raw_text) if ocr_result.raw_text else 0}")
            except Exception as e:
                logger.error(f"OCR error: {e}", exc_info=True)
                raise
            
            try:
                doc_type, confidence = classifier.predict(ocr_result.raw_text)
                logger.info(f"Detected type: {doc_type} ({confidence:.2%})")
            except Exception as e:
                logger.error(f"Classifier predict error: {e}", exc_info=True)
                raise
            
            return {
                "document_type": doc_type,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
        
        finally:
            # Cleanup temp file
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except:
                    pass
    
    @app.get("/models/status")
    async def model_status():
        """Get status of loaded models."""
        return {
            "classifier": {
                "loaded": classifier.ml_classifier.is_trained,
                "path": str(CLASSIFIER_MODEL_PATH),
            },
            "ner": {
                "loaded": ner_model.is_trained,
                "path": str(NER_MODEL_PATH),
                "num_labels": ner_model.num_labels,
            },
            "llm": {
                "available": llm_summarizer is not None,
                "fallback_available": True,
            },
        }
    
    return app


# ===========================
# Module-level app instance for uvicorn
# ===========================

app = create_app()


# ===========================
# Standalone Execution
# ===========================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
