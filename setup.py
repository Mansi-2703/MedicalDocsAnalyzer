#!/usr/bin/env python
"""
Setup Script - Medical Document Analyzer
Initializes the project environment, verifies dependencies, and validates setup.
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version (3.8+)."""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_files_exist():
    """Check essential files exist."""
    logger.info("\nChecking project files...")
    required_files = [
        "config.py",
        "main.py",
        "requirements.txt",
        "healthcare_dataset.json",
        "src/preprocessing/image_preprocessor.py",
        "src/classification/doc_classifier.py",
        "src/extraction/ner_model.py",
        "src/mapping/schema_mapper.py",
        "src/validation/rule_engine.py",
        "src/llm/summarizer.py",
        "src/api/fastapi_app.py",
        "training/train.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            logger.warning(f"✗ Missing: {file}")
        else:
            logger.info(f"✓ Found: {file}")
    
    return len(missing) == 0


def check_dependencies():
    """Check if key dependencies can be imported."""
    logger.info("\nChecking Python dependencies...")
    
    requirements = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("sklearn", "scikit-learn"),
        ("cv2", "OpenCV"),
        ("pytesseract", "pytesseract"),
    ]
    
    missing = []
    for module, name in requirements:
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            missing.append(name)
            logger.warning(f"✗ {name}")
    
    return len(missing) == 0, missing


def check_tesseract():
    """Check if Tesseract is installed."""
    logger.info("\nChecking Tesseract installation...")
    
    # Try to import and use pytesseract
    try:
        import pytesseract
        # Try to get version
        _ = pytesseract.get_tesseract_version()
        logger.info("✓ Tesseract installed and accessible")
        return True
    except Exception as e:
        logger.warning(f"✗ Tesseract error: {e}")
        logger.info("  Install: choco install tesseract (Windows) or apt-get install tesseract-ocr (Linux)")
        return False


def check_cuda():
    """Check CUDA availability."""
    logger.info("\nChecking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.info("✓ CPU mode (GPU not detected)")
            return False
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return False


def install_dependencies():
    """Install missing dependencies (optional)."""
    logger.info("\nAttempting to install dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    logger.info("\nCreating directories...")
    directories = [
        "data/raw",
        "data/processed",
        "models/ner_model",
        "models/classifier",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {directory}")


def test_imports():
    """Test that all modules can be imported."""
    logger.info("\nTesting module imports...")
    
    try:
        from src.preprocessing import ImagePreprocessor, OCREngine
        logger.info("✓ Preprocessing module")
        
        from src.classification import DocumentClassifier, EnsembleDocumentClassifier
        logger.info("✓ Classification module")
        
        from src.extraction import MedicalNERModel
        logger.info("✓ Extraction (NER) module")
        
        from src.mapping import SchemaMapper
        logger.info("✓ Mapping module")
        
        from src.validation import RuleEngine
        logger.info("✓ Validation module")
        
        from src.llm import MedicalDocumentSummarizer
        logger.info("✓ LLM module")
        
        from src.api.fastapi_app import create_app
        logger.info("✓ API module")
        
        from main import MedicalDocumentAnalysisPipeline
        logger.info("✓ Main inference pipeline")
        
        from training.train import train_all_models
        logger.info("✓ Training module")
        
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False


def verify_dataset():
    """Verify dataset structure."""
    logger.info("\nVerifying healthcare_dataset.json...")
    
    import json
    from pathlib import Path
    
    dataset_path = Path("healthcare_dataset.json")
    if not dataset_path.exists():
        logger.error("Dataset file not found")
        return False
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("Dataset must be a JSON array")
            return False
        
        if len(data) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Check first sample
        sample = data[0]
        required_fields = ["id", "document_type", "raw_text", "lines", "tokens", "ner_tags"]
        
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            logger.error(f"Missing fields in sample: {missing_fields}")
            return False
        
        logger.info(f"✓ Dataset valid: {len(data)} samples")
        logger.info(f"  Sample document types: {set(item.get('document_type') for item in data[:10])}")
        
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def print_summary(results):
    """Print setup summary."""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check}")
    
    all_pass = all(results.values())
    logger.info("=" * 60)
    
    if all_pass:
        logger.info("\n✓ All checks passed! Ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Train models:       python training/train.py healthcare_dataset.json")
        logger.info("2. Run inference:      python main.py image.jpg")
        logger.info("3. Start API server:   python -m uvicorn src.api.fastapi_app:app --reload")
    else:
        logger.info("\n✗ Some checks failed. See above for details.")
        logger.info("Run 'python setup.py --install' to install dependencies")


def main():
    """Run all checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Medical Document Analyzer")
    parser.add_argument("--install", action="store_true", help="Install missing dependencies")
    parser.add_argument("--skip-tesseract", action="store_true", help="Skip Tesseract check")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MEDICAL DOCUMENT ANALYZER - SETUP VERIFICATION")
    logger.info("=" * 60)
    
    results = {}
    
    # Run checks
    results["Python Version"] = check_python_version()
    results["Project Files"] = check_files_exist()
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    results["Python Dependencies"] = deps_ok
    
    if not deps_ok and args.install:
        logger.info("\nAttempting to install dependencies...")
        install_success = install_dependencies()
        results["Python Dependencies"] = install_success
        
        # Re-check
        deps_ok, _ = check_dependencies()
        results["Python Dependencies"] = deps_ok
    
    if not args.skip_tesseract:
        results["Tesseract"] = check_tesseract()
    
    results["Module Imports"] = test_imports()
    results["Dataset Structure"] = verify_dataset()
    results["Directories"] = True  # Always create
    
    # Create directories
    create_directories()
    
    # Check GPU (optional)
    try:
        check_cuda()
    except:
        pass
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
