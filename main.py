"""
Main Inference Pipeline - End-to-end medical document analysis.
Orchestrates all components: preprocessing, OCR, classification, NER, mapping, validation, LLM.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class InferencePipelineOutput:
    """Complete inference output."""
    request_id: str
    document_type: str
    document_type_confidence: float
    structured_output: Dict[str, Any]
    validation_results: Optional[Dict] = None
    abnormalities: Optional[list] = None
    quality_warnings: Optional[list] = None
    clinical_summary: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    warnings: list = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), default=str, indent=2)


class MedicalDocumentAnalysisPipeline:
    """
    Complete pipeline for medical document analysis.
    
    Pipeline Steps:
    1. Load and preprocess image
    2. OCR extraction with Tesseract
    3. Document type classification
    4. NER token extraction and tagging
    5. Schema mapping to structured JSON
    6. Validation with rule engine
    7. Clinical summary generation (LLM)
    8. Return final JSON response
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize pipeline with all components.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        from config import (
            PREPROCESSING_CONFIG, NER_MODEL_PATH, CLASSIFIER_MODEL_PATH,
            TAG_TO_ID, ID_TO_TAG, SCHEMA_MAPPING, VALIDATION_RULES,
            TESSERACT_PATH
        )
        from src.preprocessing import ImagePreprocessor, OCREngine
        from src.classification import EnsembleDocumentClassifier
        from src.extraction import MedicalNERModel
        from src.mapping import SchemaMapper
        from src.validation import RuleEngine
        from src.llm import MedicalDocumentSummarizer, FallbackSummarizer
        
        logger.info("Initializing pipeline components...")
        
        self.preprocessor = ImagePreprocessor(PREPROCESSING_CONFIG)
        self.ocr_engine = OCREngine(tesseract_path=TESSERACT_PATH)
        self.classifier = EnsembleDocumentClassifier()
        self.ner_model = MedicalNERModel(tag_to_id=TAG_TO_ID, id_to_tag=ID_TO_TAG)
        self.schema_mapper = SchemaMapper(SCHEMA_MAPPING)
        self.rule_engine = RuleEngine(VALIDATION_RULES)
        self.fallback_summarizer = FallbackSummarizer()
        
        # Load pre-trained models if available
        self._load_models(CLASSIFIER_MODEL_PATH, NER_MODEL_PATH)
        
        # Initialize LLM with fallback
        self.llm_summarizer = self._init_llm()
    
    def _load_models(self, classifier_path: str, ner_path: str):
        """Load pre-trained models."""
        try:
            if Path(classifier_path).exists():
                self.classifier.load(classifier_path)
                logger.info("✓ Classifier model loaded")
            else:
                logger.warning("✗ Classifier model not found - using keyword-based classification")
        except Exception as e:
            logger.warning(f"✗ Failed to load classifier: {e}")
        
        try:
            if Path(ner_path).exists():
                self.ner_model.load(ner_path)
                logger.info("✓ NER model loaded")
            else:
                logger.warning("✗ NER model not found - model needs training")
        except Exception as e:
            logger.warning(f"✗ Failed to load NER model: {e}")
    
    def _init_llm(self):
        """Initialize LLM with fallback."""
        try:
            from src.llm import MedicalDocumentSummarizer
            summarizer = MedicalDocumentSummarizer()
            logger.info("✓ LLM client initialized")
            return summarizer
        except Exception as e:
            logger.warning(f"✗ LLM initialization failed: {e} - using fallback summarizer")
            return None
    
    def process(self,
               image_path: str,
               document_type: Optional[str] = None,
               include_validation: bool = True,
               include_llm_summary: bool = True,
               return_ocr_text: bool = False,
               request_id: Optional[str] = None) -> InferencePipelineOutput:
        """
        Process medical document through complete pipeline.
        
        Args:
            image_path: Path to medical document image/PDF
            document_type: Force document type (auto-detect if None)
            include_validation: Include validation and rule checks
            include_llm_summary: Generate LLM summary
            return_ocr_text: Include raw OCR text in response
            request_id: Unique request ID
        
        Returns:
            InferencePipelineOutput with structured analysis
        """
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"[{request_id}] Starting document analysis: {Path(image_path).name}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Preprocessing
            logger.info(f"[{request_id}] [Step 1/8] Image preprocessing...")
            processed_image = self.preprocessor.preprocess_pipeline(image_path, aggressive=False)
            logger.info(f"✓ Image preprocessed: {processed_image.shape}")
            
            # Step 2: OCR
            logger.info(f"[{request_id}] [Step 2/8] OCR extraction...")
            ocr_result = self.ocr_engine.extract_structured(processed_image)
            raw_text = ocr_result.raw_text
            ocr_confidence = ocr_result.confidence
            logger.info(f"✓ OCR completed: {len(raw_text)} chars, {ocr_confidence:.1%} confidence")
            
            # Step 3: Classification
            logger.info(f"[{request_id}] [Step 3/8] Document classification...")
            detected_type, type_confidence = self.classifier.predict(raw_text)
            final_doc_type = document_type or detected_type
            logger.info(f"✓ Classified as: {final_doc_type} ({type_confidence:.1%})")
            
            # Step 4: NER Extraction
            logger.info(f"[{request_id}] [Step 4/8] Named entity recognition...")
            token_texts = [t["text"] for t in ocr_result.words] if isinstance(ocr_result.words[0], dict) else ocr_result.words
            
            predicted_tags = self.ner_model.predict(token_texts)
            logger.info(f"✓ NER completed: {len(token_texts)} tokens processed")
            
            # Step 5: Schema Mapping
            logger.info(f"[{request_id}] [Step 5/8] Schema mapping...")
            from src.extraction import extract_entities
            entities_by_type = extract_entities(token_texts, predicted_tags)
            
            structured_output = self.schema_mapper.map_to_structured(
                final_doc_type, entities_by_type, raw_text
            )
            logger.info(f"✓ Structured output generated")
            
            # Step 6: Validation
            validation_results = None
            abnormalities = None
            quality_warnings = None
            
            if include_validation:
                logger.info(f"[{request_id}] [Step 6/8] Validation & rule engine...")
                validation_results = self.rule_engine.validate(
                    final_doc_type,
                    structured_output,
                    {"gender": structured_output.get("gender")}
                )
                abnormalities = validation_results.get("abnormalities", [])
                quality_warnings = validation_results.get("quality_warnings", [])
                
                issue_count = len(abnormalities) + len(quality_warnings)
                logger.info(f"✓ Validation complete: {issue_count} issues found")
            
            # Step 7: LLM Summary
            clinical_summary = None
            if include_llm_summary:
                logger.info(f"[{request_id}] [Step 7/8] Clinical summary generation...")
                try:
                    if self.llm_summarizer:
                        clinical_summary = self.llm_summarizer.summarize(final_doc_type, structured_output)
                    else:
                        # Use fallback
                        if final_doc_type == "cbc_report":
                            clinical_summary = self.fallback_summarizer.summarize_cbc(structured_output)
                        elif final_doc_type == "prescription":
                            clinical_summary = self.fallback_summarizer.summarize_prescription(structured_output)
                        else:
                            clinical_summary = "Summary generation not supported for this document type"
                    
                    logger.info(f"✓ Summary generated ({len(clinical_summary)} chars)")
                except Exception as e:
                    logger.warning(f"✗ Summary generation failed: {e}")
                    clinical_summary = None
            
            # Step 8: Return Response
            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] [Step 8/8] Finalizing response...")
            
            output = InferencePipelineOutput(
                request_id=request_id,
                document_type=final_doc_type,
                document_type_confidence=type_confidence,
                structured_output=structured_output,
                validation_results=validation_results,
                abnormalities=abnormalities,
                quality_warnings=quality_warnings,
                clinical_summary=clinical_summary,
                ocr_text=raw_text if return_ocr_text else None,
                ocr_confidence=ocr_confidence if return_ocr_text else None,
                processing_time_ms=processing_time_ms,
                warnings=[]
            )
            
            logger.info(f"\n✓ Analysis completed in {processing_time_ms:.0f}ms")
            logger.info(f"{'='*60}\n")
            
            return output
        
        except Exception as e:
            logger.error(f"\n✗ Pipeline error: {e}", exc_info=True)
            logger.info(f"{'='*60}\n")
            raise
    
    def process_batch(self,
                     image_paths: list,
                     **kwargs) -> list:
        """Process multiple documents."""
        logger.info(f"Processing batch of {len(image_paths)} documents...")
        results = []
        
        for i, path in enumerate(image_paths, 1):
            logger.info(f"\n[Batch {i}/{len(image_paths)}] {Path(path).name}")
            try:
                result = self.process(path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append(None)
        
        logger.info(f"\nBatch processing complete: {len([r for r in results if r])}/{len(image_paths)} succeeded")
        return results


# =======================
# CLI Utilities
# =======================

def analyze_single_document(image_path: str, 
                           document_type: Optional[str] = None,
                           output_json: bool = True) -> Dict:
    """
    Command-line interface for single document analysis.
    
    Args:
        image_path: Path to medical document
        document_type: Optional document type override
        output_json: Print JSON output
    
    Returns:
        Structured analysis result
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pipeline = MedicalDocumentAnalysisPipeline()
    result = pipeline.process(
        image_path,
        document_type=document_type,
        include_validation=True,
        include_llm_summary=True,
        return_ocr_text=False,
    )
    
    if output_json:
        print(result.to_json())
    
    return result.to_dict()


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [document_type]")
        print("Example: python main.py document.jpg cbc_report")
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analyze_single_document(image_path, doc_type)
    print(json.dumps(result, indent=2, default=str))
