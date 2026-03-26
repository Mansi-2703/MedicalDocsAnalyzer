"""
VALUE NORMALIZER - Utility for normalizing medical values
"""

class ValueNormalizer:
    @staticmethod
    def normalize_age(age_str):
        try:
            return int(age_str.replace('yrs', '').replace('years', '').strip())
        except:
            return None
    
    @staticmethod
    def normalize_gender(gender_str):
        if not gender_str:
            return None
        gender_lower = gender_str.lower().strip()
        if gender_lower[0] in ['m', 'f']:
            return "Male" if gender_lower[0] == 'm' else "Female"
        if 'male' in gender_lower:
            return "Male"
        elif 'female' in gender_lower:
            return "Female"
        return None
    
    @staticmethod
    def normalize_test_value(value_str):
        """Extract numeric value from string like '9.5' or '9.5 g/dL'"""
        if not value_str:
            return None
        try:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', str(value_str).strip())
            return float(match.group(1)) if match else None
        except:
            return None
