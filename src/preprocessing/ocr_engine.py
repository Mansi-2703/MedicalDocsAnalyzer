"""
OCR Engine using Tesseract.
Extracts text and performs document structure analysis.
"""

import pytesseract
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR output."""
    raw_text: str
    lines: List[Dict]  # List of {text, coords, confidence}
    words: List[Dict]  # List of {text, coords, confidence}
    bounding_boxes: List[Tuple]  # (x, y, w, h)
    language: str = "eng"
    confidence: float = 0.0


class OCREngine:
    """Tesseract-based OCR engine optimized for medical documents."""
    
    def __init__(self, tesseract_path: str = None, languages: str = "eng"):
        """
        Initialize OCR engine.
        
        Args:
            tesseract_path: Path to tesseract executable
            languages: Languages for OCR (e.g., 'eng', 'eng+fra')
        """
        if tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_path
        
        self.languages = languages
        self.config = r'--oem 3 --psm 6'  # OEM 3: both legacy and LSTM, PSM 6: assume single column
    
    def extract_text(self, image: np.ndarray, return_confidence: bool = False) -> str:
        """
        Extract text from image using Tesseract.
        
        Args:
            image: Input image (numpy array)
            return_confidence: If True, returns (text, confidence)
        
        Returns:
            Extracted text or (text, confidence) tuple
        """
        logger.info("Running OCR extraction...")
        
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.languages,
                config=self.config
            )
            
            # Get confidence scores
            avg_confidence = 0.85  # Default reasonable confidence
            try:
                data = pytesseract.image_to_data(image, lang=self.languages, output_type='dict')
                # Try to extract confidence - different Tesseract versions have different output formats
                if 'confidence' in data:
                    confidences = [int(conf) for conf in data['confidence'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0.85
                elif 'conf' in data:
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0.85
                else:
                    logger.warning("No confidence field found in Tesseract output, using default")
                    avg_confidence = 0.85
            except Exception as conf_error:
                logger.warning(f"Could not extract confidence scores: {conf_error}, using default")
                avg_confidence = 0.85
            
            if return_confidence:
                return text, avg_confidence / 100.0 if avg_confidence > 1 else avg_confidence
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def extract_line_by_line(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text line by line with bounding boxes and confidence.
        
        Returns:
            List of dicts with: {text, bbox, confidence}
        """
        logger.info("Performing line-by-line OCR...")
        
        data = pytesseract.image_to_data(image, lang=self.languages, output_type='dict')
        
        lines = []
        current_line = ""
        current_boxes = []
        current_confs = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Valid text
                if data['line_num'][i] != (data['line_num'][i-1] if i > 0 else data['line_num'][i]):
                    # New line detected
                    if current_line:
                        bbox = self._merge_bboxes(current_boxes)
                        lines.append({
                            "text": current_line.strip(),
                            "bbox": bbox,
                            "confidence": np.mean(current_confs) if current_confs else 0.0,
                        })
                    current_line = ""
                    current_boxes = []
                    current_confs = []
                
                text = data['text'][i]
                current_line += text + " "
                current_boxes.append({
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "w": data['width'][i],
                    "h": data['height'][i],
                })
                current_confs.append(int(data['conf'][i]))
        
        # Add last line
        if current_line:
            bbox = self._merge_bboxes(current_boxes)
            lines.append({
                "text": current_line.strip(),
                "bbox": bbox,
                "confidence": np.mean(current_confs) if current_confs else 0.0,
            })
        
        logger.info(f"Extracted {len(lines)} lines")
        return lines
    
    def extract_words(self, image: np.ndarray) -> List[Dict]:
        """Extract words with bounding boxes and confidence."""
        data = pytesseract.image_to_data(image, lang=self.languages, output_type='dict')
        
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                words.append({
                    "text": data['text'][i],
                    "bbox": {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "w": data['width'][i],
                        "h": data['height'][i],
                    },
                    "confidence": int(data['conf'][i]),
                })
        
        return words
    
    def extract_structured(self, image: np.ndarray) -> OCRResult:
        """
        Full structured extraction combining all methods.
        
        Returns:
            OCRResult with raw_text, lines, words, and bounding boxes
        """
        logger.info("Performing structured OCR extraction...")
        
        raw_text, confidence = self.extract_text(image, return_confidence=True)
        lines = self.extract_line_by_line(image)
        words = self.extract_words(image)
        
        # Extract bounding boxes
        data = pytesseract.image_to_data(image, lang=self.languages, output_type='dict')
        bboxes = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                bboxes.append((
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i],
                ))
        
        result = OCRResult(
            raw_text=raw_text,
            lines=lines,
            words=words,
            bounding_boxes=bboxes,
            language=self.languages,
            confidence=confidence,
        )
        
        logger.info(f"OCR completed - Confidence: {confidence:.2f}")
        return result
    
    def _merge_bboxes(self, boxes: List[Dict]) -> Dict:
        """Merge multiple bounding boxes into a single bounding box."""
        if not boxes:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
        
        x_min = min(b["x"] for b in boxes)
        y_min = min(b["y"] for b in boxes)
        x_max = max(b["x"] + b["w"] for b in boxes)
        y_max = max(b["y"] + b["h"] for b in boxes)
        
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
        }
    
    def post_process_text(self, text: str) -> str:
        """
        Post-process OCR output to fix common errors.
        Tailored for medical documents.
        """
        # Fix common medical OCR errors
        replacements = {
            r'\bl\d\s': 'b',  # OCR confusion: '|0' -> 'O' (O-ring)
            r'(?i)hemoglobin': 'Hemoglobin',
            r'(?i)platelet': 'Platelet',
            r'(?i)glucose': 'Glucose',
            r'(?i)creatinine': 'Creatinine',
            r'(?i)bilirubin': 'Bilirubin',
            r'\s+': ' ',  # Normalize spaces
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()


@dataclass
class DocumentPage:
    """Represents a single page from a multi-page document."""
    page_num: int
    ocr_result: OCRResult
    preprocessed_image: Optional[np.ndarray] = None


class MultiPageOCRProcessor:
    """Handle multi-page document processing (PDFs)."""
    
    def __init__(self, ocr_engine: OCREngine):
        self.ocr_engine = ocr_engine
    
    def process_pdf(self, pdf_path: str, preprocessor) -> List[DocumentPage]:
        """
        Process PDF with multiple pages.
        
        Args:
            pdf_path: Path to PDF
            preprocessor: ImagePreprocessor instance
        
        Returns:
            List of DocumentPage objects
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            raise
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        pages = convert_from_path(pdf_path, dpi=300)
        results = []
        
        for page_num, page_image in enumerate(pages):
            logger.info(f"Processing page {page_num + 1}/{len(pages)}")
            
            # Convert PIL to numpy
            image = np.array(page_image)
            
            # Preprocess
            processed = preprocessor.preprocess_pipeline(None)  # Adapt for numpy input
            
            # OCR
            ocr_result = self.ocr_engine.extract_structured(processed)
            
            results.append(DocumentPage(
                page_num=page_num + 1,
                ocr_result=ocr_result,
                preprocessed_image=processed,
            ))
        
        logger.info(f"Completed PDF processing: {len(results)} pages")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    engine = OCREngine(languages="eng")
    # result = engine.extract_text(image_array)
