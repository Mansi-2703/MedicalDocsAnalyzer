"""
Rule Engine for Validation and Abnormal Value Detection.
Identifies critical values, quality warnings, and data inconsistencies.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class AbnormalityType(Enum):
    """Types of abnormalities detected."""
    CRITICAL_VALUE = "critical_value"
    OUT_OF_RANGE = "out_of_range"
    ABNORMAL_FLAG = "abnormal_flag"
    INCONSISTENT_DATA = "inconsistent_data"
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"


@dataclass
class AbnormalityAlert:
    """Alert for detected abnormality."""
    test_name: str
    value: Any
    abnormality_type: AbnormalityType
    severity: str  # "critical", "warning", "info"
    message: str
    reference_info: Optional[Dict] = None


@dataclass
class QualityWarning:
    """Data quality warning."""
    field: str
    issue: str
    confidence: float  # 0.0 - 1.0
    suggestion: str


class RuleEngine:
    """
    Rule-based validation engine for medical documents.
    Applies validation rules based on document type and standards.
    """
    
    def __init__(self, validation_rules: Dict = None):
        """
        Initialize rule engine.
        
        Args:
            validation_rules: Dict of {doc_type: {param: rules}}
        """
        self.validation_rules = validation_rules or {}
        self.abnormalities: List[AbnormalityAlert] = []
        self.quality_warnings: List[QualityWarning] = []
    
    def validate(self, 
                document_type: str,
                structured_data: Dict[str, Any],
                patient_demographics: Optional[Dict] = None) -> Dict:
        """
        Validate structured data against rules.
        
        Args:
            document_type: Type of document
            structured_data: Structured JSON output
            patient_demographics: Patient info for context (age, gender)
        
        Returns:
            Dict with {abnormalities, quality_warnings, is_valid}
        """
        logger.info(f"Validating {document_type} document")
        
        self.abnormalities = []
        self.quality_warnings = []
        
        # Check required fields
        self._validate_required_fields(document_type, structured_data)
        
        # Validate test values
        if document_type in ["cbc_report", "lft_report"]:
            self._validate_lab_values(document_type, structured_data, patient_demographics)
        
        # Validate prescription
        if document_type == "prescription":
            self._validate_prescription(structured_data)
        
        # Check data consistency
        self._check_data_quality(structured_data)
        
        result = {
            "is_valid": len(self.abnormalities) == 0,
            "abnormalities": [self._alert_to_dict(a) for a in self.abnormalities],
            "quality_warnings": [self._warning_to_dict(w) for w in self.quality_warnings],
            "total_issues": len(self.abnormalities) + len(self.quality_warnings),
        }
        
        logger.info(f"Validation complete: {result['total_issues']} issues found")
        return result
    
    def _validate_required_fields(self, document_type: str, data: Dict[str, Any]):
        """Check required fields for document type."""
        required_fields = {
            "cbc_report": ["patient_name", "patient_id", "tests"],
            "lft_report": ["patient_name", "patient_id", "tests"],
            "discharge_summary": ["patient_name", "diagnosis"],
            "prescription": ["patient_name", "medications"],
        }
        
        fields = required_fields.get(document_type, [])
        for field in fields:
            if field not in data or not data[field]:
                self.quality_warnings.append(QualityWarning(
                    field=field,
                    issue=f"Required field '{field}' is missing or empty",
                    confidence=0.95,
                    suggestion=f"Ensure {field} is extracted from document"
                ))
    
    def _validate_lab_values(self, document_type: str, 
                            data: Dict[str, Any],
                            patient_demographics: Optional[Dict] = None):
        """Validate lab test values against normal ranges."""
        rules = self.validation_rules.get(document_type, {})
        tests = data.get("tests", [])
        
        logger.info(f"Validating {len(tests)} tests for {document_type}")
        
        for test in tests:
            test_name = test.get("test_name", "").lower()
            value = test.get("value")
            
            # Skip tests with no name
            if not test_name or not test_name.strip():
                continue
            
            # Extract numeric value from string (handle "12.5 g/dL" format)
            numeric_value = None
            if value:
                # Try to extract first number from value string
                import re as regex_module
                match = regex_module.search(r'(\d+(?:\.\d+)?)', str(value))
                if match:
                    try:
                        numeric_value = float(match.group(1))
                    except (ValueError, TypeError):
                        pass
            
            if numeric_value is None:
                continue  # Skip if no valid numeric value found
            
            # Normalize test name for lookup
            normalized_name = self._normalize_test_name(test_name)
            
            logger.debug(f"Checking test: '{test_name}' (normalized: '{normalized_name}') value: {numeric_value}")
            
            if normalized_name in rules:
                rule = rules[normalized_name]
                
                # Check critical values
                if "critical_low" in rule and numeric_value < rule["critical_low"]:
                    self.abnormalities.append(AbnormalityAlert(
                        test_name=test_name,
                        value=numeric_value,
                        abnormality_type=AbnormalityType.CRITICAL_VALUE,
                        severity="critical",
                        message=f"{test_name}: CRITICAL LOW ({numeric_value})",
                        reference_info=rule,
                    ))
                    logger.info(f"CRITICAL LOW detected: {test_name} = {numeric_value}")
                
                elif "critical_high" in rule and numeric_value > rule["critical_high"]:
                    self.abnormalities.append(AbnormalityAlert(
                        test_name=test_name,
                        value=numeric_value,
                        abnormality_type=AbnormalityType.CRITICAL_VALUE,
                        severity="critical",
                        message=f"{test_name}: CRITICAL HIGH ({numeric_value})",
                        reference_info=rule,
                    ))
                    logger.info(f"CRITICAL HIGH detected: {test_name} = {numeric_value}")
                
                # Check normal range (handle both generic and gender-specific)
                normal_range = None
                if "normal_range" in rule:
                    normal_range = rule["normal_range"]
                elif patient_demographics:
                    # Try gender-specific ranges
                    gender = patient_demographics.get("gender", "").lower()
                    if gender in ["male", "m"]:
                        normal_range = rule.get("normal_range_male")
                    elif gender in ["female", "f"]:
                        normal_range = rule.get("normal_range_female")
                
                if normal_range and (numeric_value < normal_range[0] or numeric_value > normal_range[1]):
                    self.abnormalities.append(AbnormalityAlert(
                        test_name=test_name,
                        value=numeric_value,
                        abnormality_type=AbnormalityType.OUT_OF_RANGE,
                        severity="warning",
                        message=f"{test_name}: Out of normal range ({numeric_value}, normal: {normal_range[0]}-{normal_range[1]})",
                        reference_info=rule,
                    ))
                    logger.info(f"OUT OF RANGE detected: {test_name} = {numeric_value}, normal: {normal_range}")
            else:
                logger.debug(f"No validation rule found for: {normalized_name}")
    
    def _validate_prescription(self, data: Dict[str, Any]):
        """Validate prescription data."""
        rules = self.validation_rules.get("prescription", {})
        medications = data.get("medications", [])
        
        must_have = rules.get("must_have_fields", [])
        
        for i, med in enumerate(medications):
            for field in must_have:
                if not med.get(field):
                    self.quality_warnings.append(QualityWarning(
                        field=f"medication[{i}].{field}",
                        issue=f"Required medication field '{field}' is missing",
                        confidence=0.9,
                        suggestion=f"Ensure medication {i+1} has {field}"
                    ))
    
    def _check_data_quality(self, data: Dict[str, Any]):
        """Check overall data quality and consistency."""
        # Check for empty critical fields
        patient_id = data.get("patient_id")
        patient_name = data.get("patient_name")
        
        if patient_id and not re.match(r'^[A-Z]{1,3}\d{4,}$', str(patient_id).upper()):
            self.quality_warnings.append(QualityWarning(
                field="patient_id",
                issue=f"Patient ID format unusual: {patient_id}",
                confidence=0.5,
                suggestion="Verify patient ID format matches hospital standard"
            ))
        
        # Check age validity
        age = data.get("age")
        if age:
            try:
                age_int = int(age)
                if age_int < 0 or age_int > 150:
                    self.quality_warnings.append(QualityWarning(
                        field="age",
                        issue=f"Age value out of bounds: {age_int}",
                        confidence=0.8,
                        suggestion="Verify age value"
                    ))
            except (ValueError, TypeError):
                pass
    
    def _normalize_test_name(self, test_name: str) -> str:
        """Normalize test name for rule lookup."""
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', test_name.lower()).strip()
        # Replace common aliases
        aliases = {
            'hemoglobin': 'hemoglobin',
            'hb': 'hemoglobin',
            'hgb': 'hemoglobin',
            'hemoglobin g': 'hemoglobin',
            'hb g': 'hemoglobin',
            'hematocrit': 'hematocrit',
            'hct': 'hematocrit',
            'ht': 'hematocrit',
            'hematocrit pcv': 'hematocrit',
            'mcv': 'mcv',
            'mean cell volume': 'mcv',
            'mean corpuscular volume': 'mcv',
            'mch': 'mch',
            'mean cell hemoglobin': 'mch',
            'mean corpuscular hemoglobin': 'mch',
            'mchc': 'mchc',
            'mean corpuscular hemoglobin concentration': 'mchc',
            'wbc': 'wbc',
            'white blood cell': 'wbc',
            'white blood cells': 'wbc',
            'tlc': 'wbc',
            'total leucocyte': 'wbc',
            'total leucocyte count': 'wbc',
            'total wbc': 'wbc',
            'wbc count': 'wbc',
            'rbc': 'rbc',
            'red blood cell': 'rbc',
            'red blood cells': 'rbc',
            'red cell count': 'rbc',
            'rbc count': 'rbc',
            'platelets': 'platelets',
            'platelet': 'platelets',
            'plt': 'platelets',
            'platelet count': 'platelets',
            'esr': 'esr',
            'erythrocyte sedimentation rate': 'esr',
            'sed rate': 'esr',
            'neutrophil': 'neutrophil',
            'neutrophils': 'neutrophil',
            'neutrophil count': 'neutrophil',
            'lymphocyte': 'lymphocyte',
            'lymphocytes': 'lymphocyte',
            'lymphocyte count': 'lymphocyte',
            'monocyte': 'monocyte',
            'monocytes': 'monocyte',
            'monocyte count': 'monocyte',
            'eosinophil': 'eosinophil',
            'eosinophils': 'eosinophil',
            'eosinophil count': 'eosinophil',
            'basophil': 'basophil',
            'basophils': 'basophil',
            'basophil count': 'basophil',
            'bilirubin total': 'bilirubin_total',
            'total bilirubin': 'bilirubin_total',
            'bilirubin': 'bilirubin_total',
            'albumin': 'albumin',
            'serum albumin': 'albumin',
            'sgpt': 'sgpt',
            'alt': 'sgpt',
            'alanine aminotransferase': 'sgpt',
            'sgot': 'sgot',
            'ast': 'sgot',
            'aspartate aminotransferase': 'sgot',
            'alp': 'alp',
            'alkaline phosphatase': 'alp',
            'ggt': 'ggt',
            'gamma glutamyl transferase': 'ggt',
            'total proteins': 'total_proteins',
            'total protein': 'total_proteins',
            'proteins': 'total_proteins',
            'protein': 'total_proteins',
            'globulin': 'globulin',
            'serum globulin': 'globulin',
            'cholesterol': 'cholesterol',
            'cholesterol total': 'cholesterol',
            'triglycerides': 'triglycerides',
            'triglyceride': 'triglycerides',
        }
        return aliases.get(normalized, normalized)
    
    @staticmethod
    def _alert_to_dict(alert: AbnormalityAlert) -> Dict:
        """Convert AbnormalityAlert to dict."""
        return {
            "test_name": alert.test_name,
            "value": alert.value,
            "type": alert.abnormality_type.value,
            "severity": alert.severity,
            "message": alert.message,
            "reference": alert.reference_info,
        }
    
    @staticmethod
    def _warning_to_dict(warning: QualityWarning) -> Dict:
        """Convert QualityWarning to dict."""
        return {
            "field": warning.field,
            "issue": warning.issue,
            "confidence": warning.confidence,
            "suggestion": warning.suggestion,
        }


# =======================
# Specialized Validators
# =======================

class CBCValidator:
    """Specialized validator for CBC reports."""
    
    @staticmethod
    def validate_hemoglobin_consistency(value: float, 
                                       hematocrit: Optional[float] = None,
                                       gender: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate hemoglobin value and consistency with hematocrit.
        
        Rule: Hct ≈ 3 × Hemoglobin (approximately)
        """
        if not hematocrit:
            return True, ""
        
        expected_hct = value * 3
        actual_hct = hematocrit
        
        error_percent = abs(actual_hct - expected_hct) / expected_hct * 100
        
        if error_percent > 15:
            return False, f"Hemoglobin-Hematocrit mismatch: Expected Hct ~{expected_hct}, got {actual_hct}"
        
        return True, ""


class LabDataValidator:
    """General lab data validation utilities."""
    
    @staticmethod
    def is_valid_unit_for_test(test_name: str, unit: str) -> bool:
        """Check if unit is appropriate for test."""
        valid_units = {
            "hemoglobin": ["g/dL", "g/dl", "g/100ml"],
            "wbc": ["/μL", "/uL", "/ul", "x10^3/μL"],
            "platelets": ["/μL", "/uL", "x10^3/μL"],
            "bilirubin": ["mg/dL", "mg/dl"],
            "albumin": ["g/dL", "g/dl"],
        }
        
        test_normalized = test_name.lower().split()[0]
        allowed_units = valid_units.get(test_normalized, [])
        
        return any(unit.lower() in allowed_units for allowed_units in [allowed_units])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example validation rules
    from config import VALIDATION_RULES
    
    engine = RuleEngine(VALIDATION_RULES)
    
    test_data = {
        "patient_name": "Test Patient",
        "patient_id": "PAT46039",
        "tests": [
            {
                "test_name": "Hemoglobin",
                "value": "6.5",
                "unit": "g/dL",
                "reference_range": "13.0-16.0",
                "flag": "Critical Low",
            }
        ]
    }
    
    result = engine.validate("cbc_report", test_data, {"gender": "Male"})
    print(f"Validation Result: {result}")
