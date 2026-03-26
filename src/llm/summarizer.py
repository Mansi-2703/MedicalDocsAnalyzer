"""
LLM Integration for Clinical Summaries.
Generates human-readable summaries from structured JSON only (no raw text).
"""

from typing import Dict, Optional, List, Any
from enum import Enum
import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"


class LLMPromptTemplate:
    """LLM Prompt templates for different document types - Fine-tuned for medical context with patient-friendly explanations."""
    
    SYSTEM_PROMPT = """You are an expert medical document summarization AI, trained to help patients understand their medical test results in plain English.

Your role:
- Explain complex medical findings in language a non-medical person can understand
- Be empathetic and reassuring while accurate
- Focus on what the patient needs to DO, not just technical details
- Always include context about what's normal vs abnormal
- Provide actionable recommendations where appropriate
- NEVER fabricate test values or findings not in the provided data

Important rules:
1. Use lay terms with medical terms in parentheses (e.g., "hemoglobin (protein that carries oxygen)")
2. Explain severity clearly: normal, mildly abnormal, seriously abnormal, critical/urgent
3. For abnormal findings, explain the cause and implications
4. Include "normal ranges" so patient understands why it's concerning
5. Suggest what doctor might recommend next
6. ALWAYS start with an overview line so patient knows status immediately

CRITICAL DISCLAIMER TO INCLUDE AT END:
⚠️ **IMPORTANT:** This summary is educational and for informational purposes only. 
It is NOT a medical diagnosis and does NOT replace professional medical advice.
Always consult with your doctor for medical concerns, diagnosis, and treatment decisions. """ 
    
    # =======================
    # CBC Report Summary (Patient-Friendly)
    # =======================
    CBC_PROMPT_TEMPLATE = """
A patient just received their Complete Blood Count (CBC) test results. 
Please create a PATIENT-FRIENDLY summary that helps them understand their results.

**Patient:** {patient_name} ({age} year old {gender})  
**Test Date:** {report_date}

**Test Values From Lab:**
{test_results}

**Lab's Impression:**
{impression}

**Abnormal Findings:**
{abnormal_findings}

## Your task:
Write a 8-10 sentence PATIENT SUMMARY that:

1. **START with a clear status line** - Are results normal, mostly normal with minor issues, or significantly abnormal?
   Example: "Your blood work is totally normal" OR "You have mild anemia that needs monitoring"

2. **For EACH abnormal test**, explain in simple terms:
   - What was measured (e.g., "Hemoglobin measures oxygen in your blood")
   - What your value is
   - What's normal for someone your age and gender
   - Why this matters (what symptoms it might cause)
   - What might have caused it (e.g., "This often means not enough iron in diet")

3. **Include severity assessment:**
   - Green flag indicators (everything OK)
   - Yellow flags (monitor, discuss with doctor)
   - Red flags (needs urgent medical attention)

4. **Provide patient guidance:**
   - Diet/lifestyle changes that might help
   - When to contact their doctor
   - What symptoms to watch for
   - When to seek emergency care

5. **End with actionable next steps:**
   - "See your doctor for follow-up in X weeks"
   - "Take iron supplements as prescribed"
   - "Return immediately if you have difficulty breathing"

Write in a friendly, reassuring tone. This is NOT a medical diagnosis - make clear they should talk to their doctor.
Include the CRITICAL DISCLAIMER about consulting healthcare provider."""
    
    # =======================
    # LFT Report Summary (Patient-Friendly)
    # =======================
    LFT_PROMPT_TEMPLATE = """
A patient just received their Liver Function Test (LFT) results. 
Please create a PATIENT-FRIENDLY summary about their liver health.

**Patient:** {patient_name} ({age} year old {gender})  
**Test Date:** {report_date}

**LFT Values:**
{test_results}

**Lab's Impression:**
{impression}

**Abnormal Findings:**
{abnormal_findings}

## Your task:
Write a 8-10 sentence PATIENT SUMMARY that:

1. **START with liver health status:**
   Example: "Your liver is working normally" OR "We found signs that your liver might be slightly irritated"

2. **Explain the liver tests in simple terms:**
   - "Bilirubin measures a waste product your liver processes"
   - "AST/ALT are liver enzymes - high levels mean liver inflammation"
   - "Albumin is a protein your liver makes - low levels might mean liver disease"

3. **For abnormal values, explain:**
   - What's elevated/decreased
   - Why it matters for liver health
   - Common causes (hepatitis, fatty liver, medications, alcohol, etc.)
   - Severity (mild irritation vs serious liver disease)

4. **Risk assessment:**
   - Is this acute (recent injury) or chronic (long-term problem)?
   - How urgent is follow-up?
   - Danger signs to watch for (yellowing skin, dark urine, severe fatigue)

5. **Lifestyle recommendations:**
   - Alcohol and the liver
   - Medications that might affect liver
   - Dietary suggestions
   - When to avoid certain foods/drugs

6. **Next steps:**
   - What follow-up tests might be needed
   - When to see the doctor
   - Referral to specialist if needed

Write in an educational, empathetic tone. Avoid medical jargon when possible.
Include the CRITICAL DISCLAIMER."""
    
    # =======================
    # Discharge Summary (Patient-Friendly)
    # =======================
    DISCHARGE_PROMPT_TEMPLATE = """
A patient is being discharged from the hospital. They need a PATIENT-FRIENDLY summary of:
- Why they were in the hospital
- What treatment they received
- What medications to take at home
- How to care for themselves at home
- When to return to the hospital

**Patient:** {patient_name} ({age} year old {gender})  

**Primary Reason for Hospital Stay:**
{diagnosis}

**Medications to Take at Home:**
{medications}

**Doctor's Discharge Notes:**
{impression}

## Your task:
Write a 10-12 sentence PATIENT DISCHARGE SUMMARY that:

1. **Summarize the hospital stay:**
   - Why patient needed hospitalization (in simple terms)
   - Main treatments received (surgery, medication, etc.)
   - Overall condition at discharge (better, stable, needs monitoring)

2. **Medication instructions (VERY IMPORTANT):**
   - For each medication: what it's for, how much, when to take
   - Important interactions (take with/without food, don't combine with alcohol)
   - Potential side effects to expect
   - When to call if concerned about medication

3. **Home care instructions:**
   - Activity restrictions (bed rest, no heavy lifting, etc.)
   - Wound care if applicable
   - Diet restrictions or recommendations
   - When it's safe to return to work/normal activities

4. **Warning signs - When to return to hospital URGENTLY:**
   - Fever above 101°F
   - Severe pain
   - Bleeding or wound issues
   - Difficulty breathing
   - Chest pain or other emergencies

5. **Follow-up care:**
   - Appointment with primary doctor (give timeframe)
   - Specialist follow-ups if needed
   - Lab tests that need repeating
   - Physical therapy or rehabilitation

6. **Lifestyle changes:**
   - Long-term medications to continue
   - Dietary or exercise changes
   - Avoid smoking/alcohol
   - Stress management

Write CLEARLY with numbered lists for medications and warning signs.
Use all caps for URGENT WARNING SIGNS.
Include the CRITICAL DISCLAIMER."""
    
    # =======================
    # Prescription Summary (Patient-Friendly)
    # =======================
    PRESCRIPTION_PROMPT_TEMPLATE = """
A patient received a prescription and needs help understanding their medications CLEARLY.

**Patient:** {patient_name}  
**Prescription Date:** {prescription_date}

**Medications Prescribed:**
{medications}

## Your task:
Write a 6-8 sentence PATIENT MEDICATION SUMMARY that:

1. **Medication count and types:**
   "You have been prescribed 3 medications: an antibiotic, a pain reliever, and a supplement"

2. **FOR EACH MEDICATION explain (use simple language):**
   - What is it used for (disease/condition it treats)
   - Dose (strength and amount)
   - Frequency (how many times per day, morning/night)
   - Duration (how long to take it)
   - WITH OR WITHOUT food
   - Any spacing requirements (e.g., "don't take with milk")

3. **Critical interactions:**
   - Don't mix with alcohol
   - Don't take with certain other drugs
   - Take separately from other medications (time gap)
   - Food interactions (some drugs need food, others don't)

4. **Expected side effects:**
   - Common (usually OK): dizziness, nausea, dry mouth
   - Concerning (call doctor): severe allergic reaction, bleeding
   - Difference between side effects and emergencies

5. **Usage tips for success:**
   - Use pill organizer
   - Set phone alarms for doses
   - Take at same time each day
   - Don't stop abruptly without asking doctor

6. **When to talk to pharmacist/doctor:**
   - Side effects that worry you
   - If medication isn't helping
   - Before starting new medications
   - Before stopping medication

7. **Storage and safety:**
   - Keep out of reach of children/pets
   - Store in cool, dry place
   - Don't share with family members

Write in a clear, organized way with bullet points or numbering.
Avoid medical jargon. Be encouraging - most people do fine with medications.
Include the CRITICAL DISCLAIMER."""
    
    @classmethod
    def format_cbc_prompt(cls, data: Dict[str, Any]) -> str:
        """Format CBC prompt."""
        tests_str = cls._format_test_results(data.get("tests", []))
        abnormal_str = cls._format_abnormalities(data.get("abnormalities", []))
        impression = data.get("impression", "No specific impression provided")
        
        return cls.CBC_PROMPT_TEMPLATE.format(
            patient_name=data.get("patient_name", "Unknown"),
            age=data.get("age", "Unknown"),
            gender=data.get("gender", "Unknown"),
            report_date=data.get("report_date", "Unknown"),
            test_results=tests_str or "No test results available",
            impression=impression,
            abnormal_findings=abnormal_str or "No abnormal findings",
        )
    
    @classmethod
    def format_lft_prompt(cls, data: Dict[str, Any]) -> str:
        """Format LFT prompt."""
        tests_str = cls._format_test_results(data.get("tests", []))
        abnormal_str = cls._format_abnormalities(data.get("abnormalities", []))
        impression = data.get("impression", "No specific impression provided")
        
        return cls.LFT_PROMPT_TEMPLATE.format(
            patient_name=data.get("patient_name", "Unknown"),
            age=data.get("age", "Unknown"),
            gender=data.get("gender", "Unknown"),
            report_date=data.get("report_date", "Unknown"),
            test_results=tests_str or "No test results available",
            impression=impression,
            abnormal_findings=abnormal_str or "No abnormal findings",
        )
    
    @classmethod
    def format_discharge_prompt(cls, data: Dict[str, Any]) -> str:
        """Format discharge summary prompt."""
        medications_str = cls._format_medications(data.get("medications", []))
        diagnosis_str = ", ".join(data.get("diagnosis", ["Not specified"]))
        impression = data.get("clinical_impression", "No specific impression provided")
        
        return cls.DISCHARGE_PROMPT_TEMPLATE.format(
            patient_name=data.get("patient_name", "Unknown"),
            age=data.get("age", "Unknown"),
            gender=data.get("gender", "Unknown"),
            diagnosis=diagnosis_str,
            medications=medications_str or "No medications prescribed",
            impression=impression,
        )
    
    @classmethod
    def format_prescription_prompt(cls, data: Dict[str, Any]) -> str:
        """Format prescription prompt."""
        medications_str = cls._format_medications(data.get("medications", []))
        
        return cls.PRESCRIPTION_PROMPT_TEMPLATE.format(
            patient_name=data.get("patient_name", "Unknown"),
            prescription_date=data.get("prescription_date", "Unknown"),
            medications=medications_str or "No medications listed",
        )
    
    @staticmethod
    def _format_test_results(tests: List[Dict]) -> str:
        """Format test results for prompt."""
        if not tests:
            return ""
        
        lines = []
        for test in tests:
            value = test.get("value", "N/A")
            unit = test.get("unit", "")
            flag = test.get("flag", "Normal")
            ref_range = test.get("reference_range", "N/A")
            
            line = f"- {test.get('test_name', 'Unknown')}: {value} {unit} (Normal Range: {ref_range}) [{flag}]"
            lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_medications(medications: List[Dict]) -> str:
        """Format medications for prompt."""
        if not medications:
            return ""
        
        lines = []
        for med in medications:
            name = med.get("medication_name", "Unknown")
            dosage = med.get("dosage", "N/A")
            frequency = med.get("frequency", "As directed")
            duration = med.get("duration", "")
            
            line = f"- {name} {dosage} {frequency}"
            if duration:
                line += f" for {duration}"
            lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_abnormalities(abnormalities: List[Dict]) -> str:
        """Format abnormalities for prompt."""
        if not abnormalities:
            return ""
        
        lines = []
        for abn in abnormalities:
            severity = abn.get("severity", "").upper()
            message = abn.get("message", "")
            lines.append(f"[{severity}] {message}")
        
        return "\n".join(lines)


# =======================
# LLM Client Interface
# =======================

class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate_summary(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate summary from prompt."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """Initialize OpenAI client."""
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_summary(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate summary via OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLMPromptTemplate.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("google-generativeai package not installed. Install with: pip install google-generativeai")
            raise
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
    def generate_summary(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate summary via Gemini API."""
        try:
            logger.info(f"Calling Gemini API ({self.model}) for summary generation...")
            
            response = self.client.generate_content(
                f"{LLMPromptTemplate.SYSTEM_PROMPT}\n\n{prompt}",
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
            )
            
            summary = response.text
            logger.info(f"✅ Gemini summary generated successfully ({len(summary)} chars)")
            return summary
        
        except Exception as e:
            error_str = str(e)
            # Check for specific error types
            if "quota" in error_str.lower() or "429" in error_str:
                logger.error(f"❌ Gemini API QUOTA EXCEEDED: {error_str}")
                logger.info("Note: Free tier has daily quota. Consider upgrading or waiting 24 hours.")
            elif "api_key" in error_str.lower() or "unauthorized" in error_str.lower():
                logger.error(f"❌ Gemini API AUTHENTICATION FAILED: {error_str}")
                logger.info("Verify GEMINI_API_KEY in .env file")
            else:
                logger.error(f"❌ Gemini API ERROR: {error_str}")
            raise


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face Transformers client for local LLMs."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize Hugging Face client."""
        try:
            from transformers import pipeline
        except ImportError:
            logger.error("transformers package not installed. Install with: pip install transformers")
            raise
        
        self.model_name = model_name
        self.pipeline = pipeline("text-generation", model=model_name)
    
    def generate_summary(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate summary via Hugging Face."""
        try:
            result = self.pipeline(
                prompt,
                max_length=max_tokens + len(prompt.split()),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
            )
            
            return result[0]["generated_text"]
        
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            raise


# =======================
# LLM Summarizer
# =======================

class MedicalDocumentSummarizer:
    """Generates clinical summaries from structured data."""
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, 
                 provider: LLMProvider = LLMProvider.GEMINI):
        """
        Initialize summarizer.
        
        Args:
            llm_client: Custom LLM client (if None, will instantiate default)
            provider: LLM provider to use (default: Gemini)
        """
        if llm_client:
            self.llm_client = llm_client
        else:
            if provider == LLMProvider.OPENAI:
                self.llm_client = OpenAIClient()
            elif provider == LLMProvider.GEMINI:
                self.llm_client = GeminiClient()
            elif provider == LLMProvider.HUGGINGFACE:
                self.llm_client = HuggingFaceClient()
            else:
                raise ValueError(f"Unknown provider: {provider}")
    
    def summarize_cbc(self, structured_data: Dict[str, Any]) -> str:
        """Generate CBC summary."""
        prompt = LLMPromptTemplate.format_cbc_prompt(structured_data)
        return self.llm_client.generate_summary(prompt)
    
    def summarize_lft(self, structured_data: Dict[str, Any]) -> str:
        """Generate LFT summary."""
        prompt = LLMPromptTemplate.format_lft_prompt(structured_data)
        return self.llm_client.generate_summary(prompt)
    
    def summarize_discharge(self, structured_data: Dict[str, Any]) -> str:
        """Generate discharge summary."""
        prompt = LLMPromptTemplate.format_discharge_prompt(structured_data)
        return self.llm_client.generate_summary(prompt)
    
    def summarize_prescription(self, structured_data: Dict[str, Any]) -> str:
        """Generate prescription summary."""
        prompt = LLMPromptTemplate.format_prescription_prompt(structured_data)
        return self.llm_client.generate_summary(prompt)
    
    def summarize(self, document_type: str, 
                 structured_data: Dict[str, Any]) -> str:
        """
        Generate summary based on document type with detailed logging.
        
        Args:
            document_type: Type of document
            structured_data: Structured JSON output
        
        Returns:
            LLM-generated summary
        """
        logger.info(f"🤖 Generating LLM summary for document type: {document_type}")
        
        try:
            if document_type == "cbc_report":
                logger.info("  → Formatting CBC data for LLM...")
                summary = self.summarize_cbc(structured_data)
            elif document_type == "lft_report":
                logger.info("  → Formatting LFT data for LLM...")
                summary = self.summarize_lft(structured_data)
            elif document_type == "discharge_summary":
                logger.info("  → Formatting discharge data for LLM...")
                summary = self.summarize_discharge(structured_data)
            elif document_type == "prescription":
                logger.info("  → Formatting prescription data for LLM...")
                summary = self.summarize_prescription(structured_data)
            else:
                logger.warning(f"❌ No summarizer for document type: {document_type}")
                return "Summary not available for this document type"
            
            if not summary or len(summary.strip()) == 0:
                logger.warning("❌ LLM returned empty summary")
                return "Unable to generate summary"
            
            logger.info(f"✅ LLM summary generated successfully ({len(summary)} characters)")
            return summary
        
        except Exception as e:
            logger.error(f"❌ LLM summary generation failed: {type(e).__name__}: {e}")
            logger.info(f"Document type: {document_type}")
            logger.info(f"Data keys: {list(structured_data.keys())}")
            raise


# =======================
# Fallback Summarizer (without LLM)
# =======================

class FallbackSummarizer:
    """Rule-based summarizer with patient-friendly explanations."""
    
    @staticmethod
    def summarize(document_type: str, 
                 structured_data: Dict[str, Any]) -> str:
        """
        Generate patient-friendly summary using rule-based approach.
        
        Args:
            document_type: Type of document
            structured_data: Structured JSON output
        
        Returns:
            Rule-based patient-friendly summary
        """
        logger.info(f"📋 Generating fallback summary for document type: {document_type}")
        
        try:
            if document_type == "cbc_report":
                logger.info("  → Generating CBC summary (rule-based)...")
                summary = FallbackSummarizer.summarize_cbc(structured_data)
            elif document_type == "lft_report":
                logger.info("  → Generating LFT summary (rule-based)...")
                summary = FallbackSummarizer.summarize_lft(structured_data) if hasattr(FallbackSummarizer, 'summarize_lft') else "LFT summary not available"
            elif document_type == "discharge_summary":
                logger.info("  → Generating discharge summary (rule-based)...")
                summary = FallbackSummarizer.summarize_discharge(structured_data)
            elif document_type == "prescription":
                logger.info("  → Generating prescription summary (rule-based)...")
                summary = FallbackSummarizer.summarize_prescription(structured_data)
            else:
                logger.warning(f"❌ No fallback summarizer for document type: {document_type}")
                return "Summary not available for this document type"
            
            if not summary or len(summary.strip()) == 0:
                logger.warning("❌ Fallback summarizer returned empty summary")
                return "Unable to generate summary"
            
            logger.info(f"✅ Fallback summary generated successfully ({len(summary)} characters)")
            return summary
        
        except Exception as e:
            logger.error(f"❌ Fallback summary generation failed: {type(e).__name__}: {e}")
            logger.info(f"Document type: {document_type}")
            logger.info(f"Data keys: {list(structured_data.keys())}")
            raise
    
    # Patient-friendly explanations for common abnormalities
    ABNORMALITY_EXPLANATIONS = {
        "hemoglobin": {
            "low": "Your hemoglobin (oxygen-carrying protein in blood) is low. This can cause tiredness, shortness of breath, or dizziness. You may need iron supplements or further evaluation.",
            "high": "Your hemoglobin is elevated. This is less common and may indicate dehydration or other conditions. Please follow up with your doctor.",
            "critical_low": "Your hemoglobin is dangerously low - urgent medical attention may be needed.",
            "critical_high": "Your hemoglobin is dangerously high - please see your doctor immediately.",
        },
        "wbc": {
            "low": "Your white blood cell count is low. This weakens your immune system, making you more prone to infections. Avoid crowded places and sick people.",
            "high": "Your white blood cell count is elevated, suggesting your body is fighting an infection or inflammation. Rest and stay hydrated.",
            "critical_low": "Your white blood cells are critically low - urgent care needed.",
            "critical_high": "Your white blood cells are critically elevated - see your doctor immediately.",
        },
        "rbc": {
            "low": "Your red blood cell count is low (anemia), which can cause fatigue, weakness, or shortness of breath.",
            "high": "Your red blood cell count is elevated, which may indicate dehydration or other conditions.",
        },
        "platelets": {
            "low": "Your platelet count is low. You may bruise easily or have longer bleeding times. Avoid contact sports.",
            "high": "Your platelet count is elevated, which can increase blood clotting risk.",
            "critical_low": "Your platelets are critically low - seek immediate medical care.",
        },
        "bilirubin": {
            "high": "Your bilirubin is elevated, which may indicate liver issues, jaundice, or bile duct problems. This needs doctor investigation.",
        },
        "sgpt": {
            "high": "Your SGPT (liver enzyme) is elevated. This suggests liver inflammation or damage, possibly from infection, medications, or fatty liver.",
        },
        "sgot": {
            "high": "Your SGOT (liver enzyme) is elevated, similar to SGPT. This may indicate liver or muscle damage.",
        },
        "albumin": {
            "low": "Your albumin (protein) is low, suggesting nutritional deficiency or liver disease. Eat more protein-rich foods.",
        },
    }
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Extract first number from string like "6.5 g/dL"
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', value)
                return float(match.group(1)) if match else None
            except:
                return None
        return None
    
    @staticmethod
    def _get_severity_category(test_name: str, value: float) -> str:
        """Categorize test value severity for explanations."""
        test_lower = test_name.lower().replace(' ', '')
        
        # Hemoglobin thresholds
        if 'hemoglobin' in test_lower or test_lower in ['hb', 'hgb']:
            if value < 7:
                return "critical_low"
            elif value < 10:
                return "low"
            elif value > 20:
                return "critical_high"
            elif value > 17:
                return "high"
        
        # WBC thresholds
        elif test_lower in ['wbc', 'tlc']:
            if value < 2:
                return "critical_low"
            elif value < 4.5:
                return "low"
            elif value > 30:
                return "critical_high"
            elif value > 11:
                return "high"
        
        # Platelets
        elif test_lower in ['platelets', 'plt']:
            if value < 50:
                return "critical_low"
            elif value < 150:
                return "low"
            elif value > 1000:
                return "critical_high"
            elif value > 400:
                return "high"
        
        # RBC
        elif test_lower == 'rbc':
            if value < 4:
                return "low"
            elif value > 6:
                return "high"
        
        # Bilirubin
        elif 'bilirubin' in test_lower:
            if value > 1.2:
                return "high"
        
        # Liver enzymes
        elif test_lower in ['sgpt', 'alt']:
            if value > 41:
                return "high"
        elif test_lower in ['sgot', 'ast']:
            if value > 40:
                return "high"
        
        # Albumin
        elif 'albumin' in test_lower:
            if value < 3.5:
                return "low"
        
        return "normal"
    
    @staticmethod
    def _explain_abnormality(test_name: str, severity: str) -> str:
        """Get patient-friendly explanation for abnormality."""
        test_lower = test_name.lower().replace(' ', '')
        
        # Find matching explanation
        for key, explanations in FallbackSummarizer.ABNORMALITY_EXPLANATIONS.items():
            if key in test_lower:
                return explanations.get(severity, f"{test_name} is {severity}. Please consult your doctor.")
        
        # Default explanation
        severity_text = {
            "critical_low": "critically low",
            "critical_high": "critically high",
            "low": "low",
            "high": "high",
            "normal": "normal"
        }
        return f"Your {test_name} is {severity_text.get(severity, 'abnormal')}. Please discuss with your doctor."
    
    @staticmethod
    def summarize_cbc(data: Dict[str, Any]) -> str:
        """Generate patient-friendly CBC summary."""
        tests = data.get("tests", [])
        abnormalities = data.get("abnormalities", [])
        patient_name = data.get("patient_name", "Patient")
        age = data.get("age", "unknown")
        gender = data.get("gender", "unknown")
        
        if not tests:
            return "No CBC results available."
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"## Complete Blood Count (CBC) Summary for {patient_name}\n")
        summary_parts.append(f"**Patient Info:** Age {age}, {gender.capitalize()}\n")
        
        # Overall status
        abnormal_count = len(abnormalities) if abnormalities else 0
        if abnormal_count == 0:
            summary_parts.append("✅ **Overall Status:** Your blood count results are within normal limits.")
        else:
            summary_parts.append(f"⚠️ **Overall Status:** {abnormal_count} abnormal finding(s) detected that need attention.\n")
        
        # Explain specific abnormalities
        if abnormalities and isinstance(abnormalities, list):
            summary_parts.append("\n**Detailed Findings:**\n")
            for abn in abnormalities[:5]:  # Top 5 abnormalities
                try:
                    if isinstance(abn, dict):
                        test_name = abn.get('test_name', 'Unknown test')
                        value = abn.get('value', None)
                        severity = abn.get('severity', abn.get('type', 'warning'))
                        
                        if value:
                            summary_parts.append(f"\n**{test_name}:** {value}")
                            
                            # Add patient-friendly explanation
                            float_val = FallbackSummarizer._safe_float(value)
                            if float_val:
                                sev_category = FallbackSummarizer._get_severity_category(test_name, float_val)
                                explanation = FallbackSummarizer._explain_abnormality(test_name, sev_category)
                                summary_parts.append(f"\n{explanation}")
                            else:
                                msg = abn.get('message', f"{test_name} is abnormal")
                                summary_parts.append(f"\n{msg}")
                except Exception as e:
                    logger.warning(f"Error explaining abnormality: {e}")
                    continue
        
        # Recommendations
        summary_parts.append("\n\n**Recommendations:**")
        if 'hemoglobin' in str(abnormalities).lower() or 'rbc' in str(abnormalities).lower():
            summary_parts.append("- Eat iron-rich foods (spinach, red meat, beans)")
            summary_parts.append("- Rest adequately and stay hydrated")
        
        if 'wbc' in str(abnormalities).lower():
            summary_parts.append("- Avoid crowded places and sick people")
            summary_parts.append("- Practice good hygiene")
            summary_parts.append("- Get adequate sleep")
        
        if 'platelets' in str(abnormalities).lower():
            summary_parts.append("- Avoid contact sports and strenuous activities")
            summary_parts.append("- Be careful to prevent cuts and injuries")
        
        summary_parts.append("\n⚠️ **DISCLAIMER:** This summary is for informational purposes only. Please consult your healthcare provider for medical diagnosis and treatment recommendations.")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def summarize_prescription(data: Dict[str, Any]) -> str:
        """Generate patient-friendly prescription summary."""
        medications = data.get("medications", [])
        patient_name = data.get("patient_name", "Patient")
        doctor_name = data.get("doctor_name", "Your doctor")
        
        if not medications:
            return "No medications prescribed."
        
        summary_parts = []
        summary_parts.append(f"## Prescription Summary for {patient_name}\n")
        summary_parts.append(f"Prescribed by: {doctor_name}\n")
        summary_parts.append(f"\n**Total Medications:** {len(medications)}\n")
        
        summary_parts.append("\n**Medication Details:**\n")
        for i, med in enumerate(medications, 1):
            med_name = med.get("medication_name", "Unknown")
            dosage = med.get("dosage", "As directed")
            frequency = med.get("frequency", "As needed")
            duration = med.get("duration", "Continue as prescribed")
            instruction = med.get("instruction", "")
            
            summary_parts.append(f"\n{i}. **{med_name}**")
            summary_parts.append(f"   - Dosage: {dosage}")
            summary_parts.append(f"   - Frequency: {frequency}")
            summary_parts.append(f"   - Duration: {duration}")
            if instruction:
                summary_parts.append(f"   - Special instructions: {instruction}")
        
        summary_parts.append("\n\n**Important Reminders:**")
        summary_parts.append("- Take medications exactly as prescribed")
        summary_parts.append("- Do not skip doses or stop abruptly without consulting your doctor")
        summary_parts.append("- Keep medications in a safe place")
        summary_parts.append("- Mark a calendar or use a pill organizer to track doses")
        summary_parts.append("- Report any side effects to your doctor immediately")
        
        summary_parts.append("\n⚠️ **DISCLAIMER:** This summary is for informational purposes only. Please consult your healthcare provider for medical advice.")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def summarize_lft(data: Dict[str, Any]) -> str:
        """Generate patient-friendly LFT (Liver Function Test) summary."""
        tests = data.get("tests", [])
        abnormalities = data.get("abnormalities", [])
        patient_name = data.get("patient_name", "Patient")
        
        if not tests:
            return "No LFT results available."
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"## Liver Function Tests (LFT) Summary for {patient_name}\n")
        
        # Overall status
        abnormal_count = len(abnormalities) if abnormalities else 0
        if abnormal_count == 0:
            summary_parts.append("✅ **Overall Status:** Your liver function tests are within normal limits. Your liver appears to be working well.")
        else:
            summary_parts.append(f"⚠️ **Overall Status:** {abnormal_count} abnormal finding(s) detected that may indicate liver concern.\n")
        
        # Explain specific abnormalities
        if abnormalities and isinstance(abnormalities, list):
            summary_parts.append("\n**Detailed Findings:**\n")
            for abn in abnormalities[:5]:
                try:
                    if isinstance(abn, dict):
                        test_name = abn.get('test_name', 'Unknown test')
                        value = abn.get('value', None)
                        
                        if value:
                            summary_parts.append(f"\n**{test_name}:** {value}")
                            
                            # Provide context for common liver markers
                            test_lower = test_name.lower()
                            if 'bilirubin' in test_lower:
                                summary_parts.append("Bilirubin is a waste product your liver processes. High levels can lead to jaundice (yellowing of skin/eyes).")
                            elif 'sgpt' in test_lower or 'alt' in test_lower:
                                summary_parts.append("SGPT/ALT are liver enzymes. High levels suggest liver inflammation, possibly from infection, fatty liver, or medications.")
                            elif 'sgot' in test_lower or 'ast' in test_lower:
                                summary_parts.append("SGOT/AST are liver enzymes. High levels may indicate liver damage, even from muscle injury.")
                            elif 'albumin' in test_lower:
                                summary_parts.append("Albumin is a protein made by your liver. Low levels suggest poor nutrition or liver disease.")
                            elif 'alkaline' in test_lower:
                                summary_parts.append("Alkaline phosphatase is an enzyme that may be elevated with liver or bone disease.")
                            else:
                                msg = abn.get('message', f"{test_name} is abnormal")
                                summary_parts.append(msg)
                except Exception as e:
                    logger.warning(f"Error explaining abnormality: {e}")
                    continue
        
        # Recommendations
        summary_parts.append("\n\n**Lifestyle Recommendations:**")
        summary_parts.append("- Limit alcohol consumption completely (if liver damage confirmed)")
        summary_parts.append("- Avoid fatty, fried foods - eat lean proteins instead")
        summary_parts.append("- Stay hydrated with plenty of water")
        summary_parts.append("- Maintain a healthy weight")
        summary_parts.append("- Exercise regularly (30 minutes daily walks recommended)")
        summary_parts.append("- Get vaccinated if recommended for hepatitis")
        
        summary_parts.append("\n**When to Contact Your Doctor:**")
        summary_parts.append("- Yellowing of skin or eyes (jaundice)")
        summary_parts.append("- Dark-colored urine")
        summary_parts.append("- Pale or clay-colored stools")
        summary_parts.append("- Abdominal pain or swelling")
        summary_parts.append("- Persistent fatigue or weakness")
        summary_parts.append("- Easy bruising or bleeding")
        
        summary_parts.append("\n⚠️ **DISCLAIMER:** This summary is for informational purposes only. Please consult your healthcare provider for medical diagnosis and treatment.")
        
        return "\n".join(summary_parts)

    @staticmethod
    def summarize_discharge(data: Dict[str, Any]) -> str:
        """Generate patient-friendly discharge summary."""
        patient_name = data.get("patient_name", "Patient")
        diagnosis = data.get("diagnosis", [])
        medications = data.get("medications", [])
        follow_up = data.get("follow_up", "")
        
        summary_parts = []
        summary_parts.append(f"## Hospital Discharge Summary for {patient_name}\n")
        
        if diagnosis:
            summary_parts.append("**Diagnosis:**")
            for diag in diagnosis[:3]:
                summary_parts.append(f"- {diag}")
        
        if medications:
            summary_parts.append("\n**Continue These Medications at Home:**")
            for med in medications[:5]:
                if isinstance(med, dict):
                    med_name = med.get("medication_name", "Unknown")
                    summary_parts.append(f"- {med_name}")
                else:
                    summary_parts.append(f"- {med}")
        
        if follow_up:
            summary_parts.append(f"\n**Follow-up Care:**\n{follow_up}")
        
        summary_parts.append("\n⚠️ **IMPORTANT:** This summary is for informational purposes. Please consult your doctor before stopping any medications or if you experience any health changes.")
        
        return "\n".join(summary_parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example (requires OPENAI_API_KEY)
    # summarizer = MedicalDocumentSummarizer()
    # summary = summarizer.summarize_cbc({"patient_name": "Test", "tests": [...]})
