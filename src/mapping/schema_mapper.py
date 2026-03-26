"""
Schema Mapper - Intelligent Document Understanding & Analysis
Converts raw OCR text + NER entities into patient-friendly structured insights.

Key Features:
1. Document-type intelligent mapping (understands what words mean in context)
2. Abnormality detection with severity assessment
3. Value interpretation (high, low, critical, normal)
4. Relationship understanding (connects related values)
5. Structured output ready for LLM summarization
6. Patient-friendly field explanations
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =======================
# Enums & Constants
# =======================

class AbnormalitySeverity(Enum):
    """Severity levels for abnormalities."""
    NORMAL = "normal"
    WARNING = "warning"  # Mildly abnormal, should check
    CRITICAL = "critical"  # Dangerously abnormal, urgent


class TestCategory(Enum):
    """Categories of medical tests."""
    CBC = "cbc"  # Complete Blood Count
    LFT = "lft"  # Liver Function Tests
    RENAL = "renal"  # Kidney/Renal tests
    CARDIAC = "cardiac"  # Heart/Cardiac tests
    METABOLIC = "metabolic"  # Blood glucose, electrolytes
    INFECTION = "infection"  # Immune/infection markers
    COAGULATION = "coagulation"  # Bleeding/clotting tests
    UNKNOWN = "unknown"


# =======================
# Reference Ranges Database
# =======================

REFERENCE_RANGES = {
    # CBC - Complete Blood Count
    "hemoglobin": {
        "male": {"min": 13.5, "max": 17.5, "unit": "g/dL"},
        "female": {"min": 12.0, "max": 15.5, "unit": "g/dL"},
        "critical_low": 7.0,
        "critical_high": 20.0,
        "category": TestCategory.CBC,
        "explanation_low": "Low hemoglobin (anemia) - carries less oxygen in blood",
        "explanation_high": "High hemoglobin - blood is too thick/concentrated",
    },
    "wbc": {
        "normal": {"min": 4500, "max": 11000, "unit": "cells/mm³"},
        "critical_low": 2000,
        "critical_high": 30000,
        "category": TestCategory.CBC,
        "explanation_low": "Low WBC - weak immune system, prone to infections",
        "explanation_high": "High WBC - body fighting infection or inflammation",
    },
    "rbc": {
        "male": {"min": 4.5, "max": 5.9, "unit": "M/mm³"},
        "female": {"min": 4.1, "max": 5.1, "unit": "M/mm³"},
        "category": TestCategory.CBC,
        "explanation_low": "Low RBC - anemia, fatigue, shortness of breath",
        "explanation_high": "High RBC - thick blood, dehydration risk",
    },
    "platelets": {
        "normal": {"min": 150000, "max": 400000, "unit": "cells/mm³"},
        "critical_low": 50000,
        "critical_high": 1000000,
        "category": TestCategory.CBC,
        "explanation_low": "Low platelets - easy bruising, prolonged bleeding",
        "explanation_high": "High platelets - increased clotting risk",
    },
    
    # LFT - Liver Function Tests
    "bilirubin_total": {
        "normal": {"min": 0.1, "max": 1.2, "unit": "mg/dL"},
        "critical": 5.0,
        "category": TestCategory.LFT,
        "explanation_high": "High bilirubin - liver not processing waste, may cause yellowing (jaundice)",
    },
    "bilirubin_direct": {
        "normal": {"min": 0.0, "max": 0.3, "unit": "mg/dL"},
        "category": TestCategory.LFT,
        "explanation_high": "High direct bilirubin - bile duct problem",
    },
    "sgpt": {
        "normal": {"min": 7, "max": 56, "unit": "U/L"},
        "critical": 200,
        "category": TestCategory.LFT,
        "explanation_high": "High SGPT (liver enzyme) - liver inflammation from infection, medications, or fatty liver",
    },
    "sgot": {
        "normal": {"min": 10, "max": 40, "unit": "U/L"},
        "critical": 200,
        "category": TestCategory.LFT,
        "explanation_high": "High SGOT (liver enzyme) - liver or muscle damage",
    },
    "albumin": {
        "normal": {"min": 3.5, "max": 5.0, "unit": "g/dL"},
        "category": TestCategory.LFT,
        "explanation_low": "Low albumin (protein) - nutritional deficiency or liver disease",
    },
    "ggtp": {
        "normal": {"min": 0, "max": 30, "unit": "U/L"},
        "category": TestCategory.LFT,
        "explanation_high": "High GGT - liver or bone disease",
    },
    
    # Blood Glucose
    "glucose": {
        "normal_fasting": {"min": 70, "max": 100, "unit": "mg/dL"},
        "normal_random": {"max": 140, "unit": "mg/dL"},
        "diabetic": {"min": 126, "unit": "mg/dL"},
        "critical_low": 50,
        "critical_high": 400,
        "category": TestCategory.METABOLIC,
        "explanation_high": "High glucose - diabetes risk, increased infection risk",
        "explanation_low": "Low glucose - hypoglycemia, severe risk",
    },
    
    # Renal/Kidney
    "creatinine": {
        "male": {"min": 0.7, "max": 1.3, "unit": "mg/dL"},
        "female": {"min": 0.6, "max": 1.1, "unit": "mg/dL"},
        "critical": 5.0,
        "category": TestCategory.RENAL,
        "explanation_high": "High creatinine - kidney not filtering waste properly",
    },
    "urea": {
        "normal": {"min": 7, "max": 20, "unit": "mg/dL"},
        "critical": 100,
        "category": TestCategory.RENAL,
        "explanation_high": "High urea/BUN - kidney problem or dehydration",
    },
}

# Test name aliases and variations
TEST_ALIASES = {
    "hemoglobin": ["hb", "hgb", "hemoglobin", "hemo"],
    "wbc": ["wbc", "tlc", "total leukocyte", "white blood cells"],
    "rbc": ["rbc", "red blood cells"],
    "platelets": ["platelets", "plt", "thrombocytes"],
    "bilirubin": ["bilirubin", "bili"],
    "sgpt": ["sgpt", "alt", "alanine aminotransferase"],
    "sgot": ["sgot", "ast", "aspartate aminotransferase"],
    "albumin": ["albumin", "alb"],
    "glucose": ["glucose", "blood sugar", "sugar"],
    "creatinine": ["creatinine", "cr"],
    "urea": ["urea", "bun", "blood urea nitrogen"],
}
REVERSE_ALIASES = {alias: base for base, aliases in TEST_ALIASES.items() for alias in aliases}


# =======================
# Data Models
# =======================

@dataclass
class PatientDemographics:
    """Patient demographic information with defaults."""
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None  # "Male" or "Female"
    contact: Optional[str] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HospitalInfo:
    """Hospital and service provider information."""
    hospital_name: Optional[str] = None
    department: Optional[str] = None
    doctor_name: Optional[str] = None
    report_date: Optional[str] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Abnormality:
    """Detected abnormality in a test result."""
    test_name: str
    observed_value: float
    reference_range: str
    severity: AbnormalitySeverity
    direction: str  # "HIGH", "LOW", "CRITICAL"
    explanation: str  # Patient-friendly explanation
    action_item: Optional[str] = None  # What patient should do
    unit: Optional[str] = None
    
    def to_dict(self):
        return {
            "test_name": self.test_name,
            "observed_value": self.observed_value,
            "reference_range": self.reference_range,
            "severity": self.severity.value,
            "direction": self.direction,
            "explanation": self.explanation,
            "action_item": self.action_item,
            "unit": self.unit,
        }


@dataclass
class LabTestResult:
    """Single lab test result with intelligent abnormality detection."""
    test_name: str
    value: Optional[str] = None  # Raw value as string
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None
    is_abnormal: bool = False
    critical_value: bool = False
    abnormality: Optional[Abnormality] = None  # Detected abnormality details
    patient_friendly_name: Optional[str] = None  # "Hemoglobin" instead of "HB"
    test_category: Optional[TestCategory] = None
    interpretation: Optional[str] = None  # "low", "high", "normal", "critical"
    
    def to_dict(self):
        data = {
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "is_abnormal": self.is_abnormal,
            "critical_value": self.critical_value,
            "interpretation": self.interpretation,
        }
        if self.abnormality:
            data["abnormality"] = self.abnormality.to_dict()
        return data


@dataclass
class Medication:
    """Medication prescription with context."""
    medication_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None  # "oral", "injection", etc
    instruction: Optional[str] = None
    patient_friendly_usage: Optional[str] = None  # "Take with food", etc
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DocumentAnalysis:
    """Complete document analysis with understanding."""
    document_type: str
    patient_demographics: PatientDemographics
    hospital_info: HospitalInfo
    tests: List[LabTestResult] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    abnormalities: List[Abnormality] = field(default_factory=list)
    impressions: Optional[str] = None
    follow_up_recommendations: Optional[str] = None
    quality_warnings: List[str] = field(default_factory=list)
    clinical_context: Dict[str, Any] = field(default_factory=dict)  # Doctor's notes, context
    
    def to_dict(self):
        """Convert to serializable dictionary."""
        return {
            "document_type": self.document_type,
            "patient_demographics": self.patient_demographics.to_dict(),
            "hospital_info": self.hospital_info.to_dict(),
            "tests": [t.to_dict() for t in self.tests],
            "medications": [m.to_dict() for m in self.medications],
            "diagnoses": self.diagnoses,
            "abnormalities": [a.to_dict() for a in self.abnormalities],
            "impressions": self.impressions,
            "follow_up_recommendations": self.follow_up_recommendations,
            "quality_warnings": self.quality_warnings,
            "clinical_context": self.clinical_context,
        }


# =======================
# Schema Mapper
# =======================

class SchemaMapper:
    """Maps extracted NER entities to structured document-specific JSON."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def map_to_structured(self, document_type: str, 
                         entities_by_type: Dict[str, List[str]], 
                         raw_text: Optional[str] = None) -> Dict:
        """
        Map extracted entities to structured output based on document type.
        
        Args:
            document_type: Type of document (cbc_report, lft_report, etc)
            entities_by_type: Dictionary of entity type -> list of values
            raw_text: Original OCR text for fallback regex extraction
        
        Returns:
            Dictionary with structured output
        """
        if document_type == "cbc_report":
            return self._map_cbc_report(entities_by_type, raw_text)
        elif document_type == "lft_report":
            return self._map_lft_report(entities_by_type, raw_text)
        elif document_type == "discharge_summary":
            return self._map_discharge_summary(entities_by_type, raw_text)
        elif document_type == "prescription":
            return self._map_prescription(entities_by_type, raw_text)
        elif document_type == "clinical_notes":
            return self._map_clinical_notes(entities_by_type, raw_text)
        else:
            logger.warning(f"Unknown document type: {document_type}")
            return self._map_generic(entities_by_type, raw_text)
    
    def _extract_demographics(self, entities_by_type: Dict[str, List[str]]) -> PatientDemographics:
        """Extract patient demographics from entities."""
        return PatientDemographics(
            patient_name=self._get_first(entities_by_type, "PATIENT_NAME"),
            patient_id=self._get_first(entities_by_type, "PATIENT_ID"),
            age=self._get_first(entities_by_type, "AGE"),
            gender=self._get_first(entities_by_type, "GENDER"),
        )
    
    def _extract_hospital_info(self, entities_by_type: Dict[str, List[str]]) -> HospitalInfo:
        """Extract hospital information from entities."""
        return HospitalInfo(
            hospital_name=self._get_first(entities_by_type, "HOSPITAL_NAME"),
            department=self._get_first(entities_by_type, "DEPARTMENT"),
            doctor_name=self._get_first(entities_by_type, "DOCTOR_NAME"),
            report_date=self._get_first(entities_by_type, "REPORT_DATE"),
        )
    
    def _extract_lab_tests(self, entities_by_type: Dict[str, List[str]]) -> List[LabTestResult]:
        """Extract lab test results from entities."""
        tests = []
        test_names = entities_by_type.get("TEST_NAME", [])
        test_values = entities_by_type.get("TEST_VALUE", [])
        units = entities_by_type.get("UNIT", [])
        ref_ranges = entities_by_type.get("REFERENCE_RANGE", [])
        flags = entities_by_type.get("FLAG", [])
        
        # Match tests with their values
        for i, test_name in enumerate(test_names):
            test = LabTestResult(
                test_name=test_name,
                value=test_values[i] if i < len(test_values) else None,
                unit=units[i] if i < len(units) else None,
                reference_range=ref_ranges[i] if i < len(ref_ranges) else None,
                flag=flags[i] if i < len(flags) else None,
            )
            tests.append(test)
        
        return tests
    
    def _extract_medications(self, entities_by_type: Dict[str, List[str]]) -> List[Medication]:
        """Extract medications from entities."""
        medications = []
        med_names = entities_by_type.get("MEDICATION_NAME", [])
        dosages = entities_by_type.get("DOSAGE", [])
        frequencies = entities_by_type.get("FREQUENCY", [])
        durations = entities_by_type.get("DURATION", [])
        routes = entities_by_type.get("ROUTE", [])
        instructions = entities_by_type.get("INSTRUCTION", [])
        
        for i, med_name in enumerate(med_names):
            med = Medication(
                medication_name=med_name,
                dosage=dosages[i] if i < len(dosages) else None,
                frequency=frequencies[i] if i < len(frequencies) else None,
                duration=durations[i] if i < len(durations) else None,
                route=routes[i] if i < len(routes) else None,
                instruction=instructions[i] if i < len(instructions) else None,
            )
            medications.append(med)
        
        return medications
    
    def _extract_lft_tests_regex(self, text: str) -> List[LabTestResult]:
        """Extract LFT test results using regex patterns specific to liver function tests."""
        lft_patterns = [
            (r"Bilirubin\s+(?:Total|TOTAL)", r"(?:mg/dL|mg/dl|umol/L)"),
            (r"(?:Bilirubin\s+)?Direct", r"(?:mg/dL|mg/dl)"),
            (r"(?:Bilirubin\s+)?Indirect", r"(?:mg/dL|mg/dl)"),
            (r"(?:SGPT|ALT|Alanine Aminotransferase)", r"(?:IU/L|U/L|/L)"),
            (r"(?:SGOT|AST|Aspartate Aminotransferase)", r"(?:IU/L|U/L|/L)"),
            (r"(?:SGOT|AST)?/?(?:SGPT|ALT)?\s+(?:Ratio|RATIO)", r"(?:ratio|RATIO)"),
            (r"(?:Alkaline\s+)?Phosphatase", r"(?:IU/L|U/L|/L)"),
            (r"(?:GGT|Gamma\s+(?:Glutamyl\s+)?Transferase)", r"(?:IU/L|U/L|/L)"),
            (r"Total\s+(?:Proteins?|PROTEINS?)", r"(?:g/dL|g/dl|g%)"),
            (r"Albumin", r"(?:g/dL|g/dl|g%)"),
            (r"Globulin", r"(?:g/dL|g/dl|g%)"),
            (r"(?:A|a)\s*[:/]?(?:G|g)\s*(?:Ratio|RATIO)", r"(?:ratio|RATIO)"),
            (r"Cholesterol(?:\s+Total)?", r"(?:mg/dL|mg/dl)"),
            (r"Triglyceride", r"(?:mg/dL|mg/dl)"),
        ]
        
        seen_tests = set()
        tests = []
        lines = text.split('\n')
        
        for line in lines:
            if not line.strip() or len(line) < 5:
                continue
            
            if not re.search(r'\d+(?:\.\d+)?', line):
                continue
            
            for test_pattern, unit_pattern in lft_patterns:
                test_match = re.search(test_pattern, line, re.IGNORECASE)
                if not test_match:
                    continue
                
                test_name = test_match.group(0)
                test_key = test_name.lower().replace(' ', '').replace('\t', '')
                
                if test_key in seen_tests:
                    continue
                
                # Extract numeric value
                value = None
                value_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if value_match:
                    value = value_match.group(1)
                
                # Extract unit
                unit = None
                unit_match = re.search(unit_pattern, line, re.IGNORECASE)
                if unit_match:
                    unit = unit_match.group(0)
                
                # Extract reference range
                ref_range = None
                ref_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', line)
                if ref_match:
                    ref_range = f"{ref_match.group(1)} - {ref_match.group(2)}"
                
                if test_name and (value or ref_range):
                    test = LabTestResult(
                        test_name=test_name,
                        value=value,
                        unit=unit,
                        reference_range=ref_range,
                        flag=None,
                    )
                    tests.append(test)
                    seen_tests.add(test_key)
                    break
        
        return tests
    
    def _extract_medications_regex(self, text: str) -> List[Medication]:
        """Extract medications from raw text using regex patterns."""
        medications = []
        seen_meds = set()
        
        # Patterns to match medication entries
        # Pattern 1: "Tab Medication_Name dosage frequency"
        # Pattern 2: "Medication_Name dose frequency"
        # Pattern 3: Lines with common medication indicators
        
        med_patterns = [
            # Tab/Cap/Inj Medication dosage frequency
            r'(?:Tab|Cap|Inj|Syrup|Ointment|Cream|Suspension)\.?\s+([A-Za-z0-9\s\-]+?)\s+(\d+\s*(?:mg|ml|g|mIU)?)\s+(.*?)(?=(?:Tab|Cap|Inj|$|\n))',
            # Medication with dosage pattern: "Medication_Name dose x frequency"
            r'([A-Z][A-Za-z\s]+?)\s+(\d+\s*(?:mg|ml|g|mcg|IU)?)\s+([A-Za-z0-9\s\-]+?)(?:\n|$)',
            # "- Medication dosage frequency"
            r'[-•]\s+([A-Za-z\s\-]+?)\s+(\d+[\s\-]*(?:mg|ml|g|mcg|IU)?)\s+(.*?)(?=\n[-•]|$)',
        ]
        
        for pattern in med_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                if len(match.groups()) >= 1:
                    med_name = match.group(1).strip() if match.group(1) else None
                    dosage = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else None
                    frequency = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else None
                    
                    # Validate medication name
                    if med_name and 2 < len(med_name) < 100:
                        med_key = med_name.lower().replace(' ', '')
                        if med_key not in seen_meds and not any(skip in med_name.lower() for skip in ['patient', 'for', 'and', 'the', 'with']):
                            med = Medication(
                                medication_name=med_name,
                                dosage=dosage,
                                frequency=frequency,
                                route=None,
                                duration=None,
                                instruction=None,
                            )
                            medications.append(med)
                            seen_meds.add(med_key)
        
        return medications
    
    def _extract_discharge_summary_regex(self, text: str) -> Dict[str, Any]:
        """Extract discharge summary content (diagnosis, medications, treatments) from raw text."""
        result = {
            "diagnosis": [],
            "treatments": [],
            "medications": [],
            "follow_up": None,
            "advice": None,
        }
        
        # Extract diagnosis section
        diagnosis_patterns = [
            r"(?:Primary\s+)?Diagnosis[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|Medications|Treatment|Follow|Course)",
            r"(?:Clinical\s+)?Condition[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|$)",
            r"(?:Chief\s+)?Complaint[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|$)",
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                diagnosis_text = match.group(1).strip()
                # Split by commas or "and" to get individual diagnoses
                diagnoses = [d.strip() for d in re.split(r',|and', diagnosis_text) if d.strip()]
                result["diagnosis"] = diagnoses
                break
        
        # Extract treatment/course section
        treatment_patterns = [
            r"(?:Course\s+in\s+)?Hospital[:\s]+([A-Za-z\s\-,\.]+?)(?:\n\s*$|\nMedications|\nMedicine|\nFollow)",
            r"Treatment\s+Given[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|Follow|Advice)",
            r"Management[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|Follow|Advice)",
        ]
        
        for pattern in treatment_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                treatment_text = match.group(1).strip()
                # Split into separate treatments/procedures
                treatments = [t.strip() for t in re.split(r',|;', treatment_text) if t.strip() and len(t.strip()) > 3]
                result["treatments"] = treatments
                break
        
        # Extract medications used during hospitalization
        result["medications"] = self._extract_medications_regex(text)
        
        # Extract follow-up advice
        followup_patterns = [
            r"Follow.?up[:\s]+([^\n]+?)(?:\n|$)",
            r"Advice[:\s]+([^\n]+?)(?:\n|$)",
            r"Return\s+to\s+(?:clinic|hospital)[:\s]+([^\n]+?)(?:\n|$)",
        ]
        
        for pattern in followup_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result["follow_up"] = match.group(1).strip()
                break
        
        # Extract general advice section
        advice_patterns = [
            r"Advice[:\s]+([A-Za-z\s\-,\.]+?)(?:\n\s*$|Follow|Return)",
            r"Instructions[:\s]+([A-Za-z\s\-,\.]+?)(?:\n|$)",
        ]
        
        for pattern in advice_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result["advice"] = match.group(1).strip()
                break
        
        return result
    
    def _extract_clinical_notes_regex(self, text: str) -> Dict[str, Any]:
        """Extract clinical notes content (complaints, examination, assessment, plan) from raw text."""
        result = {
            "presenting_problem": None,
            "vital_signs": {},
            "examination": None,
            "assessment": None,
            "plan": None,
            "medications": [],
        }
        
        # Extract presenting problem/chief complaint
        complaint_patterns = [
            r"(?:Presenting\s+)?Problem[:\s]+([A-Za-z\s,.\-]+?)(?:\n|Vital|Examination|Assessment)",
            r"Chief\s+Complaint[:\s]+([A-Za-z\s,.\-]+?)(?:\n|Vital|Examination)",
            r"Patient\s+(?:Reports?\s+)?(?:Feeling|Complains)[:\s]+([A-Za-z\s,.\-]+?)(?:\n|$)",
        ]
        
        for pattern in complaint_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result["presenting_problem"] = match.group(1).strip()
                break
        
        # Extract vital signs
        vital_signs_patterns = {
            "blood_pressure": r"(?:BP|Blood\s+Pressure)[:\s]+(\d+/\d+)\s*(?:mmHg)?",
            "pulse_rate": r"(?:PR|Pulse\s+Rate)[:\s]+(\d+)\s*(?:bpm)?",
            "spo2": r"(?:SPO2|O2\s+Sat)[:\s]+(\d+)\s*(?:%)?",
            "temperature": r"(?:Temp|Temperature)[:\s]+(\d+\.?\d*)\s*(?:°[CF]|Celsius|Fahrenheit)?",
            "respiratory_rate": r"(?:RR|Respiratory\s+Rate)[:\s]+(\d+)\s*(?:bpm)?",
        }
        
        for vital_name, pattern in vital_signs_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["vital_signs"][vital_name] = match.group(1).strip()
        
        # Extract examination findings
        exam_patterns = [
            r"(?:Physical\s+)?Examination[:\s]+([A-Za-z\s,.\-]+?)(?:\n\s*(?:Assessment|Plan|Impression)|\n\s*$)",
            r"Clinical\s+Examination[:\s]+([A-Za-z\s,.\-]+?)(?:\n|Assessment|Plan)",
        ]
        
        for pattern in exam_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                exam_text = match.group(1).strip()
                # Clean up multiline examination
                exam_text = ' '.join(exam_text.split())
                result["examination"] = exam_text if len(exam_text) > 10 else None
                break
        
        # Extract assessment/diagnosis
        assessment_patterns = [
            r"(?:Diagnostic\s+)?Impression[:\s]*([A-Za-z\s,.\-]+?)(?=\n(?:Plan|Intervention|Follow)|$)",
            r"Assessment[:\s]+([A-Za-z\s,.\-]+?)(?:\n(?:Plan|Intervention)|$)",
            r"Diagnosis[:\s]+([A-Za-z\s,.\-]+?)(?:\n|Plan)",
        ]
        
        for pattern in assessment_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result["assessment"] = match.group(1).strip()
                break
        
        # Extract treatment plan
        plan_patterns = [
            r"(?:Treatment\s+)?Plan[:\s]+([A-Za-z\s,.\-]+?)(?=\n\s*(?:Follow|Return|Medications)|$)",
            r"Plan\s+for\s+(?:Next\s+)?(?:Session|Visit)[:\s]+([A-Za-z\s,.\-]+?)(?:\n|$)",
            r"Interventions[:\s]+([A-Za-z\s,.\-]+?)(?=\n\s*(?:Follow|Client)|\n\s*$)",
        ]
        
        for pattern in plan_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                plan_text = match.group(1).strip()
                result["plan"] = ' '.join(plan_text.split()) if len(plan_text) > 10 else None
                break
        
        # Extract any medications mentioned
        result["medications"] = self._extract_medications_regex(text)
        
        return result
    
    def _extract_patient_name_regex(self, text: str) -> Optional[str]:
        """Extract patient name from raw text using flexible patterns for all document types."""
        if not text:
            return None
        
        # More specific and filtered patterns
        patterns = [
            # Explicit "Patient Name:" or "Name:" patterns
            r"(?:Patient\s+)?Name[:\s]+([A-Z][A-Za-z\s\.]+?)(?:\n|Age|DOB|Gender|Sex|M/F|ID|MRN|Ref)",
            # "Pt Name" variant
            r"Pt\.?\s+Name[:\s]+([A-Z][A-Za-z\s\.]+?)(?:\n|Age|$)",
            # Name after specific markers (not generic lines)
            r"(?:Patient|Pt)\s+Name[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = ' '.join(name.split())  # Clean whitespace
                
                # Validation: filter out headers, invalid patterns, etc.
                if (len(name) > 3 and len(name) < 100 and  # Reasonable length
                    not any(skip in name.lower() for skip in ['leading', 'the', 'way', 'patient', 'report', 'lab', 'analysis']) and
                    name.count(' ') < 5 and  # Max 5 parts (reasonable for names)
                    not name.replace(' ', '').isdigit()):  # Not just numbers
                    return name
        
        return None
    
    def _extract_hospital_name_regex(self, text: str) -> Optional[str]:
        """Extract hospital/lab name from raw text."""
        if not text:
            return None
        
        patterns = [
            r"(?:Hospital|Centre|Center|Lab|Laboratory|Clinic|Diagnostic\.?|Path\.?)[:\s]+([A-Za-z\s\.,&]+?)(?:\n|Address|Tel|Phone|Email)",
            r"(?:Ref|Reference)(?:\s+Lab|oratory)?[:\s]+([A-Za-z\s\.,&]+?)(?:\n|$)",
            r"^([A-Za-z\s\.,&]{5,50})\s*(?:Hospital|Centre|Center|Lab|Laboratory|Clinic)",  # Name before hospital/lab
        ]
        
        for pattern in patterns:
            if pattern.startswith("^"):
                for line in text.split('\n'):
                    if match := re.match(pattern, line.strip(), re.IGNORECASE):
                        hosp = match.group(1).strip()
                        if 3 < len(hosp) < 100:
                            return hosp
            else:
                if match := re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    hosp = match.group(1).strip()
                    if 3 < len(hosp) < 100:
                        return hosp
        
        return None
    
    def _extract_patient_id_regex(self, text: str) -> Optional[str]:
        """Extract patient ID/MRN from raw text using flexible regex."""
        if not text:
            return None
        
        patterns = [
            # Standard patterns with explicit labels (highest priority)
            (r"(?:Patient\s+)?ID[:\s]+([A-Z0-9\-\.]+)(?:\s|$|\n|Age)", True),  # Require alphanumeric
            (r"MRN[:\s]+([A-Z0-9\-\.]+)(?:\s|$|\n)", True),
            (r"(?:Ref(?:erence)?|Reg|OP)\s*(?:No\.?|Number)[:\s]+([A-Z0-9\-\.]+)(?:\s|$|\n)", True),
            # Pattern for ID with more specific format (must have digits)
            (r"(?:Patient\s+)?ID[:\s]*([A-Z]?[-\.]?\d{3,}[-\.]?\d*)", True),
            # Lab ID pattern
            (r"Lab(?:oratory)?\s+ID[:\s]+([A-Z0-9\-\.]+)", True),
        ]
        
        for pattern, require_numeric in patterns:
            if pattern.startswith("^"):
                for line in text.split('\n'):
                    if re.match(pattern, line.strip()):
                        patient_id = line.strip()
                        if self._is_valid_patient_id(patient_id, require_numeric):
                            return patient_id
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    patient_id = match.group(1).strip()
                    if self._is_valid_patient_id(patient_id, require_numeric):
                        return patient_id
        
        return None
    
    @staticmethod
    def _is_valid_patient_id(patient_id: str, require_numeric: bool = True) -> bool:
        """Validate if a string looks like a patient ID."""
        if not patient_id or not (2 < len(patient_id) < 50):
            return False
        
        # Blocklist of known non-ID values
        blocklist = ['age', 'gender', 'date', 'name', 'id', 'mrn', 'patient', 'ref', 'range',
                    'hate', 'salf', 'self', 'male', 'female', 'report', 'lab', 'analysis',
                     'test', 'result', 'unit', 'normal', 'abnormal']
        
        if patient_id.lower() in blocklist:
            return False
        
        # Must have at least one alphanumeric
        if not any(c.isalnum() for c in patient_id):
            return False
        
        # If numeric is required, must have digits
        if require_numeric and not any(c.isdigit() for c in patient_id):
            return False
        
        # Should not be all letters (unless it's a known format like "PN2", "RE1")
        if patient_id.isalpha() and len(patient_id) > 3:
            return False
        
        return True
    
    def _extract_age_regex(self, text: str) -> Optional[str]:
        """Extract age from raw text using regex."""
        if not text:
            return None
        
        patterns = [
            r"Age[:\s]+(\d+)\s*(?:y(?:ears?)?|yrs?)",
            r"Age[:\s]+(\d+)",
            r"(\d+)\s*y(?:ears?)?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = match.group(1).strip()
                age_num = ValueNormalizer.normalize_age(age)
                if age_num and 0 < age_num < 150:  # Reasonable age
                    return str(age_num)
        
        return None
    
    def _extract_gender_regex(self, text: str) -> Optional[str]:
        """Extract gender from raw text using regex."""
        if not text:
            return None
        
        patterns = [
            r"(?:Sex|Gender)[:\s]+([MFmf]\.?)",
            r"(?:Sex|Gender)[:\s]+([Mm]ale|[Ff]emale)",
            r"[Mm]/[Ff][:\s]+([MFmf])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).strip()
                return ValueNormalizer.normalize_gender(gender)
        
        return None
    
    def _extract_date_regex(self, text: str) -> Optional[str]:
        """Extract report/test date from raw text using regex."""
        if not text:
            return None
        
        patterns = [
            r"(?:Report|Test)\s*Date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?:Collected|Sample)\s*(?:Date|Time)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_cbc_tests_regex(self, text: str) -> List[LabTestResult]:
        """Extract CBC tests from raw text using regex patterns for table format."""
        tests = []
        if not text:
            return tests
        
        # CBC test patterns with comprehensive variations and their typical units
        cbc_patterns = [
            (r"(?:Total\s+)?(?:Leucocytes?|WBC|White\s+Blood\s+Cells?|Total\s+RBC)", r"/[μu]L|K/μL|K/uL|cells/μL|x10\^3"),
            (r"Hemoglobin|Hb|HGB", r"g/d[Ll]|g/100\s*ml|gm%"),
            (r"Hematocrit|HCT|Ht\.", r"%"),
            (r"(?:Red\s+Blood\s+Cells?|RBC)", r"M/μL|M/uL|million|x10\^6|m/[μu]L"),
            (r"MCV|Mean\s+Cell\s+Volume", r"f[Ll]|fL|femtoliters?"),
            (r"MCH\b|Mean\s+Cell\s+Hemoglobin\b", r"pg|picograms?"),
            (r"MCHC|Mean\s+Corpuscular\s+Hemoglobin", r"g/d[Ll]|%|gm%"),
            (r"(?:Platelet|PLT|Thrombocytes?)", r"/[μu]L|K/μL|K/uL|x10\^3|thousand"),
            (r"ESR|Erythrocyte\s+Sedimentation\s+Rate", r"mm/h|mm/hr"),
            (r"Neutrophil", r"%"),
            (r"Lymphocyte", r"%"),
            (r"Monocyte", r"%"),
            (r"Eosinophil", r"%"),
            (r"Basophil", r"%"),
        ]
        
        seen_tests = set()  # Track tests to avoid duplicates
        lines = text.split('\n')
        
        for line in lines:
            # Skip empty or short lines
            if not line.strip() or len(line) < 5:
                continue
            
            # Look for lines with numeric values
            if not re.search(r'\d+(?:\.\d+)?', line):
                continue
            
            # Try to match against CBC patterns
            for test_pattern, unit_pattern in cbc_patterns:
                test_match = re.search(test_pattern, line, re.IGNORECASE)
                if not test_match:
                    continue
                
                test_name = test_match.group(0)
                
                # Skip duplicates
                test_key = test_name.lower().replace(' ', '').replace('\t', '')
                if test_key in seen_tests:
                    continue
                
                # Extract first numeric value on this line
                value = None
                value_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if value_match:
                    value = value_match.group(1)
                
                # Extract unit - search more broadly
                unit = None
                # First try specific pattern
                unit_match = re.search(unit_pattern, line, re.IGNORECASE)
                if unit_match:
                    unit = unit_match.group(0)
                else:
                    # Fallback: look for any unit-like pattern
                    unit_fallback = re.search(r'([a-zA-Z%/]+(?:\^?\d+)?(?:[/\^\-]?[a-zA-Z]+)?)', line)
                    if unit_fallback:
                        potential_unit = unit_fallback.group(0)
                        # Filter out common non-unit words
                        if potential_unit.lower() not in ['and', 'the', 'ref', 'range']:
                            unit = potential_unit
                
                # Extract reference range (number - number)
                ref_range = None
                ref_match = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', line)
                if ref_match:
                    ref_range = f"{ref_match.group(1)} - {ref_match.group(2)}"
                
                # Only add if we have test name and (value or ref range)
                if test_name and (value or ref_range):
                    test = LabTestResult(
                        test_name=test_name,
                        value=value,
                        unit=unit,
                        reference_range=ref_range,
                        flag=None,
                    )
                    tests.append(test)
                    seen_tests.add(test_key)
                    break  # Move to next line
        
        return tests
    
    # ===========================
    # Document-Specific Mapping
    # ===========================
    
    def _map_cbc_report(self, entities_by_type: Dict[str, List[str]], 
                       raw_text: Optional[str] = None) -> Dict:
        """Map CBC report entities to structured output with fallback regex extraction."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        tests = self._extract_lab_tests(entities_by_type)
        impression = self._get_first(entities_by_type, "CLINICAL_IMPRESSION")
        
        # Fallback to regex extraction from raw text if NER entities are empty
        if raw_text:
            if not demographics.patient_name:
                demographics.patient_name = self._extract_patient_name_regex(raw_text)
            if not demographics.patient_id:
                demographics.patient_id = self._extract_patient_id_regex(raw_text)
            if not demographics.age:
                demographics.age = self._extract_age_regex(raw_text)
            if not demographics.gender:
                demographics.gender = self._extract_gender_regex(raw_text)
            if not hospital_info.report_date:
                hospital_info.report_date = self._extract_date_regex(raw_text)
            if not hospital_info.hospital_name:
                hospital_info.hospital_name = self._extract_hospital_name_regex(raw_text)
            
            # If no tests extracted via NER, try regex extraction
            if not tests:
                tests = self._extract_cbc_tests_regex(raw_text)
        
        return {
            "hospital_name": hospital_info.hospital_name,
            "department": hospital_info.department,
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "age": demographics.age,
            "gender": demographics.gender,
            "report_date": hospital_info.report_date,
            "tests": [asdict(t) for t in tests],
            "impression": impression or "",
        }
    
    def _map_lft_report(self, entities_by_type: Dict[str, List[str]], 
                       raw_text: Optional[str] = None) -> Dict:
        """Map LFT report entities to structured output with fallback regex extraction."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        tests = self._extract_lab_tests(entities_by_type)
        impression = self._get_first(entities_by_type, "CLINICAL_IMPRESSION")
        
        # Fallback to regex extraction from raw text if NER entities are empty
        if raw_text:
            if not demographics.patient_name:
                demographics.patient_name = self._extract_patient_name_regex(raw_text)
            if not demographics.patient_id:
                demographics.patient_id = self._extract_patient_id_regex(raw_text)
            if not demographics.age:
                demographics.age = self._extract_age_regex(raw_text)
            if not demographics.gender:
                demographics.gender = self._extract_gender_regex(raw_text)
            if not hospital_info.report_date:
                hospital_info.report_date = self._extract_date_regex(raw_text)
            if not hospital_info.hospital_name:
                hospital_info.hospital_name = self._extract_hospital_name_regex(raw_text)
            
            # If no tests extracted via NER, try LFT-specific regex extraction
            if not tests:
                tests = self._extract_lft_tests_regex(raw_text)
        
        return {
            "hospital_name": hospital_info.hospital_name,
            "department": hospital_info.department,
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "age": demographics.age,
            "gender": demographics.gender,
            "report_date": hospital_info.report_date,
            "tests": [asdict(t) for t in tests],
            "impression": impression or "",
        }
    
    def _map_discharge_summary(self, entities_by_type: Dict[str, List[str]], 
                              raw_text: Optional[str] = None) -> Dict:
        """Map discharge summary entities to structured output with fallback extraction."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        medications = self._extract_medications(entities_by_type)
        diagnosis = entities_by_type.get("DIAGNOSIS", [])
        impression = self._get_first(entities_by_type, "CLINICAL_IMPRESSION")
        
        # Fallback to regex extraction from raw text if NER entities are empty
        if raw_text:
            if not demographics.patient_name:
                demographics.patient_name = self._extract_patient_name_regex(raw_text)
            if not demographics.patient_id:
                demographics.patient_id = self._extract_patient_id_regex(raw_text)
            if not demographics.age:
                demographics.age = self._extract_age_regex(raw_text)
            if not demographics.gender:
                demographics.gender = self._extract_gender_regex(raw_text)
            if not hospital_info.hospital_name:
                hospital_info.hospital_name = self._extract_hospital_name_regex(raw_text)
            if not hospital_info.report_date:
                hospital_info.report_date = self._extract_date_regex(raw_text)
            
            # Use comprehensive discharge summary regex extraction if needed
            discharge_data = self._extract_discharge_summary_regex(raw_text)
            
            # Merge regex-extracted data with NER data
            if not diagnosis:
                diagnosis = discharge_data.get("diagnosis", [])
            if not medications:
                medications = [Medication(medication_name=m) for m in discharge_data.get("medications", [])]
            if not impression:
                impression = discharge_data.get("follow_up") or discharge_data.get("advice")
        
        return {
            "hospital_name": hospital_info.hospital_name,
            "department": hospital_info.department,
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "age": demographics.age,
            "gender": demographics.gender,
            "report_date": hospital_info.report_date,
            "diagnosis": diagnosis,
            "medications": [asdict(m) for m in medications],
            "clinical_impression": impression or "",
        }
    
    def _map_prescription(self, entities_by_type: Dict[str, List[str]], 
                         raw_text: Optional[str] = None) -> Dict:
        """Map prescription entities to structured output with fallback extraction."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        medications = self._extract_medications(entities_by_type)
        
        # Fallback to regex extraction from raw text if NER entities are empty
        if raw_text:
            if not demographics.patient_name:
                demographics.patient_name = self._extract_patient_name_regex(raw_text)
            if not demographics.patient_id:
                demographics.patient_id = self._extract_patient_id_regex(raw_text)
            if not hospital_info.hospital_name:
                hospital_info.hospital_name = self._extract_hospital_name_regex(raw_text)
            if not hospital_info.report_date:
                hospital_info.report_date = self._extract_date_regex(raw_text)
            
            # If no medications extracted via NER, try regex extraction
            if not medications:
                medications = self._extract_medications_regex(raw_text)
        
        return {
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "doctor_name": hospital_info.doctor_name,
            "hospital_name": hospital_info.hospital_name,
            "prescription_date": hospital_info.report_date,
            "medications": [asdict(m) for m in medications],
        }
    
    def _map_clinical_notes(self, entities_by_type: Dict[str, List[str]], 
                           raw_text: Optional[str] = None) -> Dict:
        """Map clinical notes entities to structured output with fallback extraction."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        
        # Initialize with basic demographics fallback
        if raw_text:
            if not demographics.patient_name:
                demographics.patient_name = self._extract_patient_name_regex(raw_text)
            if not demographics.patient_id:
                demographics.patient_id = self._extract_patient_id_regex(raw_text)
            if not demographics.age:
                demographics.age = self._extract_age_regex(raw_text)
            if not demographics.gender:
                demographics.gender = self._extract_gender_regex(raw_text)
            if not hospital_info.hospital_name:
                hospital_info.hospital_name = self._extract_hospital_name_regex(raw_text)
            if not hospital_info.report_date:
                hospital_info.report_date = self._extract_date_regex(raw_text)
            
            # Extract clinical notes specific content
            notes_data = self._extract_clinical_notes_regex(raw_text)
        else:
            notes_data = {
                "presenting_problem": None,
                "vital_signs": {},
                "examination": None,
                "assessment": None,
                "plan": None,
                "medications": [],
            }
        
        return {
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "age": demographics.age,
            "gender": demographics.gender,
            "hospital_name": hospital_info.hospital_name,
            "provider_name": hospital_info.doctor_name,
            "session_date": hospital_info.report_date,
            "presenting_problem": notes_data.get("presenting_problem"),
            "vital_signs": notes_data.get("vital_signs", {}),
            "examination": notes_data.get("examination"),
            "assessment": notes_data.get("assessment"),
            "plan": notes_data.get("plan"),
            "medications": [asdict(m) for m in notes_data.get("medications", [])],
        }
    
    def _map_generic(self, entities_by_type: Dict[str, List[str]], 
                    raw_text: Optional[str] = None) -> Dict:
        """Generic mapping for unknown document types."""
        demographics = self._extract_demographics(entities_by_type)
        hospital_info = self._extract_hospital_info(entities_by_type)
        
        return {
            "document_type": "unknown",
            "hospital_name": hospital_info.hospital_name,
            "patient_name": demographics.patient_name,
            "patient_id": demographics.patient_id,
            "extracted_entities": entities_by_type,
        }
    
    @staticmethod
    def _get_first(entities: Dict[str, List], key: str, default=None) -> Any:
        """Get first value for a key or default."""
        values = entities.get(key, [])
        return values[0] if values else default


# =======================
# Field Value Normalizers
# =======================

class ValueNormalizer:
    """Normalize extracted field values."""
    
    @staticmethod
    def normalize_gender(value: str) -> str:
        """Normalize gender values."""
        if not value:
            return None
        value = value.lower().strip()
        if value in ['m', 'male', 'mr.']:
            return 'Male'
        elif value in ['f', 'female', 'mrs.', 'ms.']:
            return 'Female'
        else:
            return value.title()
    
    @staticmethod
    def normalize_age(value: str) -> int:
        """Extract and normalize age."""
        if not value:
            return None
        matches = re.findall(r'\d+', str(value))
        return int(matches[0]) if matches else None
    
    @staticmethod
    def normalize_date(value: str) -> str:
        """Normalize date format (keep as extracted, can extend)."""
        return value
    
    @staticmethod
    def normalize_numeric_value(value: str) -> float:
        """Extract numeric value from text (e.g., "13.5 g/dL" -> 13.5)."""
        if not value:
            return None
        matches = re.findall(r'\d+(\.\d+)?', str(value))
        return float(matches[0]) if matches else None
    
    @staticmethod
    def normalize_unit(value: str) -> str:
        """Normalize units."""
        if not value:
            return None
        
        unit_map = {
            '/ul': '/μL',
            '/µl': '/μL',
            'g/dl': 'g/dL',
            'mg/dl': 'mg/dL',
            'u/l': 'U/L',
            'meq/l': 'mEq/L',
        }
        
        normalized = value.lower().strip()
        return unit_map.get(normalized, value)
    
    @staticmethod
    def normalize_reference_range(value: str) -> str:
        """Normalize reference range format."""
        if not value:
            return None
        # Could implement standardization logic here
        return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mapper = SchemaMapper()
    
    # Example
    entities = {
        "PATIENT_NAME": ["Anjali Joshi"],
        "PATIENT_ID": ["PAT46039"],
        "AGE": ["72"],
        "GENDER": ["M"],
        "HOSPITAL_NAME": ["Apollo Healthcare"],
        "TEST_NAME": ["Hemoglobin", "WBC"],
        "TEST_VALUE": ["13.5", "7500"],
        "UNIT": ["g/dL", "/uL"],
    }
    
    result = mapper.map_to_structured("cbc_report", entities)
    print(result)
