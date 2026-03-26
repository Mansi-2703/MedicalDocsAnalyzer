"""
DOCUMENT UNDERSTANDING ENGINE - Orchestrates document analysis
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

from src.mapping.abnormality_detector import AbnormalityDetector, AbnormalityFlag, AbnormalitySeverity
from src.mapping.value_normalizer import ValueNormalizer

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    CBC = "cbc"
    LFT = "lft"
    METABOLIC = "metabolic"
    RENAL = "renal"
    CARDIAC = "cardiac"
    UNKNOWN = "unknown"


@dataclass
class PatientDemographics:
    """Patient information"""
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    contact: Optional[str] = None


@dataclass
class HospitalInfo:
    """Healthcare provider information"""
    hospital_name: Optional[str] = None
    department: Optional[str] = None
    doctor_name: Optional[str] = None
    report_date: Optional[str] = None


@dataclass
class LabTestResult:
    """Lab test with abnormality information"""
    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: bool = False
    abnormality_flag: Optional[AbnormalityFlag] = None
    category: Optional[TestCategory] = None

    def to_dict(self):
        data = {
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "is_abnormal": self.is_abnormal,
            "category": self.category.value if self.category else None,
        }
        if self.abnormality_flag:
            data["abnormality"] = self.abnormality_flag.to_dict()
        return data


@dataclass
class Medication:
    """Medication with usage information"""
    medication_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instruction: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DocumentAnalysis:
    """Complete intelligent document analysis"""
    document_type: str
    patient_demographics: PatientDemographics
    hospital_info: HospitalInfo
    tests: List[LabTestResult] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    abnormalities: List[AbnormalityFlag] = field(default_factory=list)
    impressions: Optional[str] = None
    follow_up: Optional[str] = None
    quality_warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        """Convert to serializable dictionary"""
        return {
            "document_type": self.document_type,
            "patient_demographics": asdict(self.patient_demographics),
            "hospital_info": asdict(self.hospital_info),
            "tests": [t.to_dict() for t in self.tests],
            "medications": [m.to_dict() for m in self.medications],
            "diagnoses": self.diagnoses,
            "abnormalities": [a.to_dict() for a in self.abnormalities],
            "impressions": self.impressions,
            "follow_up": self.follow_up,
           "quality_warnings": self.quality_warnings,
        }


class DocumentUnderstandingEngine:
    """Main engine for intelligent document analysis"""
    
    def __init__(self):
        self.abnormality_detector = AbnormalityDetector()
        self.normalizer = ValueNormalizer()
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, 
                document_type: str,
                entities_by_type: Dict[str, List[str]],
                raw_text: Optional[str] = None,
                patient_gender: Optional[str] = None) -> DocumentAnalysis:
        """
        Analyze a medical document intelligently
        
        Args:
            document_type: Type of document (cbc_report, lft_report, discharge_summary, prescription)
            entities_by_type: Extracted NER entities
            raw_text: Original OCR text
            patient_gender: Patient gender for gender-specific reference ranges
            
        Returns:
            Comprehensive DocumentAnalysis with abnormality detection
        """
        logger.info(f"📊 Analyzing {document_type} document...")
        
        # Extract demographics and hospital info
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        
        # Fallback to raw text extraction if needed
        if not demographics.patient_name and raw_text:
            demographics.patient_name = self._extract_patient_name(raw_text)
        if not demographics.patient_id and raw_text:
            demographics.patient_id = self._extract_patient_id(raw_text)
        if not demographics.age and raw_text:
            demographics.age = self._extract_age(raw_text)
        if not demographics.gender and raw_text:
            demographics.gender = self._extract_gender(raw_text)
        if not hospital_info.hospital_name and raw_text:
            hospital_info.hospital_name = self._extract_hospital_name(raw_text)
        
        # Normalize gender for reference range lookup
        normalized_gender = demographics.gender or patient_gender
        
        # Extract tests and detect abnormalities
        tests = self._extract_tests(entities_by_type, document_type)
        abnormalities = []
        
        for test in tests:
            # Check for abnormalities
            abnormality_flag = self.abnormality_detector.detect(
                test_name=test.test_name,
                value_str=test.value,
                gender=normalized_gender
            )
            if abnormality_flag:
                test.abnormality_flag = abnormality_flag
                test.is_abnormal = True
                abnormalities.append(abnormality_flag)
                logger.warning(f"🚩 ABNORMALITY DETECTED: {test.test_name} = {test.value} ({abnormality_flag.direction})")
        
        # Extract medications
        medications = self._extract_medications(entities_by_type)
        
        # Extract diagnoses
        diagnoses = entities_by_type.get("DIAGNOSIS", [])
        
        # Log abnormalities summary
        if abnormalities:
            logger.warning(f"🚩 {len(abnormalities)} abnormalities found in {document_type}")
            for abn in abnormalities:
                logger.warning(f"   - {abn.test_name}: {abn.direction} ({abn.severity.value})")
        else:
            logger.info(f"✅ No abnormalities detected in {document_type}")
        
        # Create analysis
        analysis = DocumentAnalysis(
            document_type=document_type,
            patient_demographics=demographics,
            hospital_info=hospital_info,
            tests=tests,
            medications=medications,
            diagnoses=diagnoses,
            abnormalities=abnormalities,
            impressions=entities_by_type.get("CLINICAL_IMPRESSION", [None])[0],
        )
        
        return analysis
    
    def _extract_demographics(self, entities_by_type: Dict[str, List[str]]) -> PatientDemographics:
        """Extract patient demographics from NER entities"""
        return PatientDemographics(
            patient_name=self._get_first(entities_by_type, "PATIENT_NAME"),
            patient_id=self._get_first(entities_by_type, "PATIENT_ID"),
            age=self._get_first(entities_by_type, "AGE"),
            gender=self._get_first(entities_by_type, "GENDER"),
        )
    
    def _extract_hospital_info(self, entities_by_type: Dict[str, List[str]]) -> HospitalInfo:
        """Extract hospital information from NER entities"""
        return HospitalInfo(
            hospital_name=self._get_first(entities_by_type, "HOSPITAL_NAME"),
            department=self._get_first(entities_by_type, "DEPARTMENT"),
            doctor_name=self._get_first(entities_by_type, "DOCTOR_NAME"),
            report_date=self._get_first(entities_by_type, "REPORT_DATE"),
        )
    
    def _extract_tests(self, entities_by_type: Dict[str, List[str]], doc_type: str) -> List[LabTestResult]:
        """Extract lab tests from entities and categorize them"""
        tests = []
        test_names = entities_by_type.get("TEST_NAME", [])
        test_values = entities_by_type.get("TEST_VALUE", [])
        units = entities_by_type.get("UNIT", [])
        ref_ranges = entities_by_type.get("REFERENCE_RANGE", [])
        
        for i, test_name in enumerate(test_names):
            category = self._categorize_test(test_name, doc_type)
            test = LabTestResult(
                test_name=test_name,
                value=test_values[i] if i < len(test_values) else None,
                unit=units[i] if i < len(units) else None,
                reference_range=ref_ranges[i] if i < len(ref_ranges) else None,
                category=category,
            )
            tests.append(test)
        
        return tests
    
    def _extract_medications(self, entities_by_type: Dict[str, List[str]]) -> List[Medication]:
        """Extract medications from entities"""
        medications = []
        med_names = entities_by_type.get("MEDICATION_NAME", [])
        dosages = entities_by_type.get("DOSAGE", [])
        frequencies = entities_by_type.get("FREQUENCY", [])
        durations = entities_by_type.get("DURATION", [])
        
        for i, med_name in enumerate(med_names):
            med = Medication(
                medication_name=med_name,
                dosage=dosages[i] if i < len(dosages) else None,
                frequency=frequencies[i] if i < len(frequencies) else None,
                duration=durations[i] if i < len(durations) else None,
            )
            medications.append(med)
        
        return medications
    
    def _categorize_test(self, test_name: str, doc_type: str) -> TestCategory:
        """Categorize a test based on name and document type"""
        test_lower = test_name.lower()
        
        cbc_keywords = ["hemoglobin", "wbc", "rbc", "platelet", "hb", "hgb", "tlc"]
        lft_keywords = ["bilirubin", "sgpt", "sgot", "albumin", "liver", "ggtp"]
        metabolic_keywords = ["glucose", "sugar", "cholesterol", "triglyceride"]
        renal_keywords = ["creatinine", "urea", "bun", "kidney"]
        
        if any(kw in test_lower for kw in cbc_keywords):
            return TestCategory.CBC
        elif any(kw in test_lower for kw in lft_keywords):
            return TestCategory.LFT
        elif any(kw in test_lower for kw in metabolic_keywords):
            return TestCategory.METABOLIC
        elif any(kw in test_lower for kw in renal_keywords):
            return TestCategory.RENAL
        
        # Fall back to document type
        if doc_type == "cbc_report":
            return TestCategory.CBC
        elif doc_type == "lft_report":
            return TestCategory.LFT
        
        return TestCategory.UNKNOWN
    
    # ===== Raw Text Extraction Fallback Methods =====
    
    def _extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name from raw text"""
        import re
        pattern = r"(?:Patient\s+)?Name[:\s]+([A-Z][A-Za-z\s\.]+?)(?:\n|Age|DOB|Gender|Sex)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if 3 < len(name) < 100:
                return name
        return None
    
    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID from raw text"""
        import re
        pattern = r"(?:Patient\s+)?ID[:\s]+([A-Z0-9\-\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_age(self, text: str) -> Optional[str]:
        """Extract age from raw text"""
        import re
        pattern = r"Age[:\s]+(\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from raw text"""
        import re
        pattern = r"(?:Sex|Gender)[:\s]+([MFmf](?:ale)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gender_str = match.group(1).lower()
            return "Male" if gender_str[0] == 'm' else "Female"
        return None
    
    def _extract_hospital_name(self, text: str) -> Optional[str]:
        """Extract hospital name from raw text"""
        import re
        # Try first line as hospital name
        first_line = text.split('\n')[0].strip() if text else None
        if first_line and len(first_line) > 5 and len(first_line) < 100:
            # Check if it looks like a hospital name (not technical data)
            if not any(x in first_line.lower() for x in ['test', 'report', 'result', 'value']):
                return first_line
        return None
    
    @staticmethod
    def _get_first(entities: Dict[str, List[str]], key: str) -> Optional[str]:
        """Get first value from entity list"""
        values = entities.get(key, [])
        return values[0] if values else None
