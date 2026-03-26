"""
ABNORMALITY DETECTOR - Intelligent detection of abnormal lab values
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AbnormalitySeverity(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AbnormalityFlag:
    """Detected abnormality details"""
    test_name: str
    observed_value: float
    reference_range: str
    severity: AbnormalitySeverity
    direction: str  # HIGH, LOW, CRITICAL
    explanation: str  # Patient-friendly
    action_item: Optional[str] = None
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


class AbnormalityDetector:
    """Detects abnormalities by comparing values to reference ranges"""
    
    # Reference ranges for common tests
    REFERENCE_RANGES = {
        "hemoglobin": {
            "male": {"min": 13.5, "max": 17.5, "unit": "g/dL"},
            "female": {"min": 12.0, "max": 15.5, "unit": "g/dL"},
            "critical_low": 7.0,
            "critical_high": 20.0,
        },
        "wbc": {
            "normal": {"min": 4500, "max": 11000, "unit": "cells/mm³"},
            "critical_low": 2000,
            "critical_high": 30000,
        },
        "rbc": {
            "male": {"min": 4.5, "max": 5.9, "unit": "M/mm³"},
            "female": {"min": 4.1, "max": 5.1, "unit": "M/mm³"},
        },
        "platelets": {
            "normal": {"min": 150000, "max": 400000, "unit": "cells/mm³"},
            "critical_low": 50000,
        },
        "bilirubin": {
            "normal": {"min": 0.1, "max": 1.2, "unit": "mg/dL"},
            "critical": 5.0,
        },
        "sgpt": {
            "normal": {"min": 7, "max": 56, "unit": "U/L"},
            "critical": 200,
        },
        "sgot": {
            "normal": {"min": 10, "max": 40, "unit": "U/L"},
            "critical": 200,
        },
        "albumin": {
            "normal": {"min": 3.5, "max": 5.0, "unit": "g/dL"},
        },
        "glucose": {
            "normal_fasting": {"min": 70, "max": 100, "unit": "mg/dL"},
            "normal_random": {"max": 140, "unit": "mg/dL"},
            "critical_low": 50,
            "critical_high": 400,
        },
    }
    
    EXPLANATIONS = {
        "hemoglobin": {
            "low": "Low hemoglobin (anemia) means your blood carries less oxygen, causing tiredness and shortness of breath.",
            "high": "High hemoglobin means blood is too thick - increased clotting risk.",
            "critical_low": "Dangerously low hemoglobin - needs urgent treatment.",
        },
        "wbc": {
            "low": "Low WBC (white blood cells) weakens immune system - avoid infections.",
            "high": "High WBC means body is fighting infection or inflammation.",
        },
        "rbc": {
            "low": "Low RBC (red blood cells) causes anemia - fatigue, weakness.",
            "high": "High RBC means blood is thick - dehydration risk.",
        },
        "bilirubin": {
            "high": "High bilirubin (liver waste) - may cause yellowing of skin/eyes.",
        },
        "sgpt": {
            "high": "High SGPT liver enzyme - inflammation from infection, fatty liver, or medications.",
        },
        "albumin": {
            "low": "Low albumin (protein) - nutrition problem or liver disease.",
        },
        "glucose": {
            "high": "High blood sugar - diabetes risk, reduced immunity.",
            "low": "Low blood sugar - emergency! Needs immediate treatment.",
            "critical_low": "Dangerously low glucose - life-threatening.",
        },
    }
    
    @staticmethod
    def normalize_test_name(test_name: str) -> str:
        """Normalize test names to lowercase for matching"""
        if not test_name:
            return ""
        normalized = test_name.lower().replace(" ", "").replace(".", "")
        # Common aliases
        aliases = {
            "hb": "hemoglobin", "hgb": "hemoglobin",
            "tlc": "wbc", "wbc": "wbc",
            "sgpt": "sgpt", "alt": "sgpt",
            "sgot": "sgot", "ast": "sgot",
            "bili": "bilirubin",
        }
        return aliases.get(normalized, normalized)
    
    @staticmethod
    def extract_numeric_value(value_str: str) -> Optional[float]:
        """Extract numeric value from formatted string"""
        if not value_str:
            return None
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(value_str).strip())
        return float(match.group(1)) if match else None
    
    def detect(self, test_name: str, value_str: str, gender: Optional[str] = None) -> Optional[AbnormalityFlag]:
        """Detect if a test value is abnormal"""
        if not test_name or not value_str:
            return None
        
        norm_test = self.normalize_test_name(test_name)
        value = self.extract_numeric_value(value_str)
        
        if value is None:
            return None
        
        ranges = self.REFERENCE_RANGES.get(norm_test)
        if not ranges:
            return None
        
        # Get appropriate range based on gender
        ref_min, ref_max = None, None
        critical_low, critical_high = None, None
        unit = "unit"
        
        if "male" in ranges or "female" in ranges:
            gender_key = "male" if gender and "male" in gender.lower() else "female"
            if gender_key in ranges:
                ref_min = ranges[gender_key].get("min")
                ref_max = ranges[gender_key].get("max")
                unit = ranges[gender_key].get("unit", "unit")
        elif "normal" in ranges:
            ref_min = ranges["normal"].get("min")
            ref_max = ranges["normal"].get("max")
            unit = ranges["normal"].get("unit", "unit")
        
        critical_low = ranges.get("critical_low")
        critical_high = ranges.get("critical_high")
        reference_range_str = f"{ref_min} - {ref_max} {unit}" if ref_min and ref_max else "N/A"
        
        # Detect abnormality
        if critical_low and value < critical_low:
            return AbnormalityFlag(
                test_name=test_name,
                observed_value=value,
                reference_range=reference_range_str,
                severity=AbnormalitySeverity.CRITICAL,
                direction="CRITICAL",
                explanation=self.EXPLANATIONS.get(norm_test, {}).get("critical_low", f"{test_name} is dangerously low"),
                action_item="SEEK IMMEDIATE MEDICAL CARE",
                unit=unit,
            )
        elif critical_high and value > critical_high:
            return AbnormalityFlag(
                test_name=test_name,
                observed_value=value,
                reference_range=reference_range_str,
                severity=AbnormalitySeverity.CRITICAL,
                direction="CRITICAL",
                explanation=self.EXPLANATIONS.get(norm_test, {}).get("critical_high", f"{test_name} is dangerously high"),
                action_item="SEEK IMMEDIATE MEDICAL CARE",
                unit=unit,
            )
        elif ref_min and value < ref_min:
            return AbnormalityFlag(
                test_name=test_name,
                observed_value=value,
                reference_range=reference_range_str,
                severity=AbnormalitySeverity.WARNING,
                direction="LOW",
                explanation=self.EXPLANATIONS.get(norm_test, {}).get("low", f"{test_name} is below normal"),
                action_item=f"Discuss with your doctor about treatment options for low {test_name}",
                unit=unit,
            )
        elif ref_max and value > ref_max:
            return AbnormalityFlag(
                test_name=test_name,
                observed_value=value,
                reference_range=reference_range_str,
                severity=AbnormalitySeverity.WARNING,
                direction="HIGH",
                explanation=self.EXPLANATIONS.get(norm_test, {}).get("high", f"{test_name} is above normal"),
                action_item=f"Consult doctor about elevated {test_name}",
                unit=unit,
            )
        
        return None  # Normal
