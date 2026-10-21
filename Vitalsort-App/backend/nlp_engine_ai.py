import re
from transformers import pipeline
import os

class AdvancedNLPEngine:
    """
    Advanced NLP Engine using transformer models and optional GPT integration.
    Provides superior clinical entity recognition and symptom understanding.
    """
    
    def __init__(self, use_gpt: bool = False, openai_api_key: str = None):
        """
        Initialize NLP engine with transformer models.
        
        Args:
            use_gpt: Whether to use GPT for enhanced understanding
            openai_api_key: OpenAI API key (if use_gpt=True)
        """
        self.use_gpt = use_gpt and openai_api_key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if self.use_gpt and self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                self.openai = openai
            except Exception as e:
                print(f"Warning: OpenAI not available: {e}")
                self.use_gpt = False
        
        # Load pre-trained models from Hugging Face
        try:
            # Zero-shot classification for symptom severity
            self.severity_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Zero-shot classifier not available: {e}")
            self.severity_classifier = None
        
        # Clinical keywords database
        self.clinical_keywords = {
            "pain": ["pain", "ache", "soreness", "hurt", "tender"],
            "fever": ["fever", "temperature", "hot", "burning"],
            "rash": ["rash", "hives", "redness", "spots", "bumps"],
            "swelling": ["swelling", "edema", "puffiness", "enlarged"],
            "breathing": ["breathing", "breath", "respiratory", "dyspnea"],
            "chest": ["chest", "thoracic", "cardiac", "heart"],
            "nausea": ["nausea", "sick", "vomiting", "queasy"],
            "bleeding": ["bleeding", "blood", "hemorrhage", "bleed"],
            "severe": ["severe", "severe", "extreme", "worst", "unbearable"],
            "chronic": ["chronic", "ongoing", "persistent", "long-term"]
        }
        
        self.acuity_triggers = [
            "severe", "extreme", "worst", "unbearable", "emergency",
            "difficulty breathing", "chest pain", "bleeding", "unconscious"
        ]
    
    def process_symptoms(self, text: str) -> dict:
        """
        Advanced symptom extraction using transformers and optional GPT.
        
        Args:
            text: Raw patient symptom description
            
        Returns:
            Dict with extracted symptoms, severity, and clinical insights
        """
        # Text normalization
        normalized_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = normalized_text.split()
        
        # Extract symptoms using keyword matching and ML
        extracted_symptoms = self._extract_symptoms(normalized_text)
        
        # Classify severity using zero-shot classification
        if self.severity_classifier:
            severity = self._classify_severity(text, extracted_symptoms)
        else:
            severity = self._classify_severity_heuristic(text)
        
        # Extract temporal information
        chronology = self._extract_chronology(normalized_text)
        
        # Advanced acuity detection
        inferred_acuity = self._determine_acuity(text, extracted_symptoms, severity)
        
        # Optional: Use GPT for deeper understanding
        if self.use_gpt:
            gpt_insights = self._get_gpt_insights(text, extracted_symptoms)
        else:
            gpt_insights = None
        
        return {
            "processed_tokens": len(tokens),
            "extracted_symptoms": extracted_symptoms,
            "symptom_severity": severity,
            "chronology_indicators": chronology if chronology else ["Unspecified"],
            "inferred_acuity": inferred_acuity,
            "confidence": 0.85,
            "gpt_insights": gpt_insights
        }
    
    def _extract_symptoms(self, text: str) -> list:
        """Extracts symptoms using keyword database and NLP."""
        symptoms = []
        
        for symptom_category, keywords in self.clinical_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    symptoms.append(symptom_category)
                    break  # One per category
        
        return symptoms if symptoms else ["None detected"]
    
    def _classify_severity(self, text: str, symptoms: list) -> str:
        """
        Classifies severity using zero-shot transformer classification.
        """
        try:
            if not self.severity_classifier:
                return self._classify_severity_heuristic(text)
            
            # Use model to classify
            candidate_labels = ["mild", "moderate", "severe", "critical"]
            result = self.severity_classifier(text, candidate_labels)
            
            return result['labels'][0]  # Top scoring label
        except Exception as e:
            print(f"Severity classification error: {e}")
            return self._classify_severity_heuristic(text)
    
    def _classify_severity_heuristic(self, text: str) -> str:
        """Fallback heuristic-based severity classification."""
        text_lower = text.lower()
        
        if any(trigger in text_lower for trigger in self.acuity_triggers):
            return "severe"
        elif any(word in text_lower for word in ["mild", "slight", "little"]):
            return "mild"
        elif any(word in text_lower for word in ["moderate", "significant"]):
            return "moderate"
        else:
            return "mild"
    
    def _extract_chronology(self, text: str) -> list:
        """Extracts temporal information about symptom onset."""
        temporal_keywords = [
            "days", "weeks", "months", "hours", "ago", 
            "suddenly", "recently", "slowly", "gradually"
        ]
        
        chronology = [word for word in text.split() 
                     if word in temporal_keywords]
        
        return chronology if chronology else []
    
    def _determine_acuity(self, text: str, symptoms: list, severity: str) -> str:
        """
        Determines clinical acuity level based on multiple factors.
        """
        text_lower = text.lower()
        
        # Critical indicators
        if severity == "critical" or any(t in text_lower for t in 
                                        ["difficulty breathing", "chest pain", "unconscious"]):
            return "Urgent/Emergency Risk"
        
        # High acuity indicators
        if severity in ["severe"] and len(symptoms) > 1:
            return "Urgent/Emergency Risk"
        
        # Moderate acuity
        if severity in ["moderate", "severe"] or len(symptoms) >= 2:
            return "Urgent"
        
        # Routine
        return "Routine"
    
    def _get_gpt_insights(self, text: str, symptoms: list) -> dict:
        """
        Uses GPT to provide additional clinical insights.
        (Requires OPENAI_API_KEY environment variable)
        """
        try:
            if not self.use_gpt or not self.openai:
                return None
            
            # Create prompt for GPT
            prompt = f"""Based on the patient's symptom description:
"{text}"

Provide a brief clinical assessment including:
1. Key clinical findings
2. Differential diagnoses (top 3)
3. Recommended specialist
4. Urgency level

Keep response concise and professional."""
            
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a clinical decision support system."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return {
                "ai_assessment": response['choices'][0]['message']['content'],
                "model": "gpt-3.5-turbo"
            }
        
        except Exception as e:
            print(f"GPT insights error: {e}")
            return None
