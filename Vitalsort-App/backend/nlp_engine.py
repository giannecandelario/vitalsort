import re

class NLPEngine:
    def __init__(self):
        """
        Initializes the Native NLP Engine.
        In a production environment, you would initialize the BioBERT 
        transformers pipeline here. For the MVP, we use foundational 
        clinical entity extraction to maintain a lightweight architecture.
        """
        self.clinical_keywords = [
            "pain", "fever", "rash", "swelling", "redness", 
            "bleeding", "nausea", "dizziness", "lesion", "itchy"
        ]
        self.temporal_keywords = [
            "days", "weeks", "months", "hours", "ago", "suddenly", "recently"
        ]
        self.acuity_triggers = [
            "severe", "extreme", "worst", "unbearable", "breathing", "chest"
        ]

    def process_symptoms(self, text: str) -> dict:
        """
        Executes text normalization, lemmatization simulation, and 
        Clinical Named Entity Recognition (NER).
        """
        # 1. Text Normalization (Lowercase, remove punctuation)
        normalized_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = normalized_text.split()

        # 2. Clinical Entity Extraction
        extracted_symptoms = [word for word in tokens if word in self.clinical_keywords]
        
        # 3. Temporal/Chronology Extraction
        chronology = [word for word in tokens if word in self.temporal_keywords]

        # 4. High-Acuity Detection (Early Red Flag routing)
        severity = "Routine"
        if any(word in tokens for word in self.acuity_triggers):
            severity = "Urgent/Emergency Risk"

        return {
            "processed_tokens": len(tokens),
            "extracted_symptoms": extracted_symptoms if extracted_symptoms else ["None detected"],
            "chronology_indicators": chronology if chronology else ["Unspecified"],
            "inferred_acuity": severity
        }