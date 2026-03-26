"""
Init file for mapping module.
"""

# Import from old schema mapper (still in use)
from .schema_mapper import (
    SchemaMapper,
    PatientDemographics,
    HospitalInfo,
    LabTestResult,
    Medication,
    DocumentAnalysis,
    AbnormalitySeverity,
    TestCategory,
)

# Import from new intelligent modules
from .value_normalizer import ValueNormalizer
from .abnormality_detector import AbnormalityDetector, AbnormalityFlag, AbnormalitySeverity as DetectorSeverity
from .document_understanding import DocumentUnderstandingEngine

__all__ = [
    # Old schema mapper (still available for backward compatibility)
    "SchemaMapper",
    "PatientDemographics",
    "HospitalInfo",
    "LabTestResult",
    "Medication",
    "DocumentAnalysis",
    "AbnormalitySeverity",
    "TestCategory",
    
    # New intelligent modules
    "ValueNormalizer",
    "AbnormalityDetector",
    "AbnormalityFlag",
    "DocumentUnderstandingEngine",
]
