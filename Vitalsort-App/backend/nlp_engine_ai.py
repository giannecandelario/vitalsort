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
        #
        # Matching is word anchored and allows one inflection (see
        # _keyword_pattern), which is what makes short stems safe: "hot" reaches
        # "hot"/"hots" but never "photo" or "hotel", "ear" never
        # reaches "hear" or "barely", "numb" never reaches "number", "pee" never
        # reaches "peeled". Irregular forms have to be listed themselves -
        # "swollen", "allergies", "blacked out".
        #
        # Category names are what the score table, the red-flag triggers and the
        # specialist matrix match against, so a name must stay stable and be
        # weighted in engines.SYMPTOM_WEIGHTS (guarded by test_triage_scoring.py).
        self.clinical_keywords = {
            "pain": ["pain", "painful", "ache", "ach", "sore", "soreness", "hurt",
                     "tender", "tenderness", "sharp"],
            "headache": ["headache", "head ache", "migraine", "head pain", "heads hurting"],
            "thunderclap headache": ["worst headache", "baddest headache", "sudden severe headache",
                                     "worst headache of my life", "explosive headache"],
            "fever": ["fever", "feverish", "temperature", "hot", "burning", "chill",
                      "night sweats", "high temp", "running a temperature"],
            "rash": ["rash", "hives", "redness", "spot", "bump", "itch", "itchy",
                     "itching", "flaky", "scaly", "breakout", "blister", "poison ivy"],
            "swelling": ["swelling", "swell", "swollen", "edema", "puffiness", "puffy",
                         "enlarged", "water retention"],
            "breathing": ["breathing", "breath", "respiratory", "dyspnea", "wheeze",
                          "wheezing", "wheezy", "gasping", "gasp", "suffocate", "suffocating",
                          "cant breathe", "can not breathe", "hard to breathe",
                          "short of breath", "air hunger", "choking", "choked", "choke"],
            "chest": ["chest", "thoracic", "cardiac", "heart"],
            "cough": ["cough", "hacking up", "phlegm", "chest infection"],
            "palpitations": ["palpitation", "racing heart", "heart pounding", "flutter",
                             "fast heartbeat", "irregular heartbeat", "skipped beat"],
            "nausea": ["nausea", "nauseous", "sick", "sickness", "vomiting", "vomit",
                       "queasy", "throwing up", "threw up", "food down", "bilious"],
            "diarrhea": ["diarrhea", "diarrhoea", "loose stool", "watery stool",
                         "bowels running", "the runs", "dysentery"],
            "constipation": ["constipation", "constipated", "blocked bowels", "no bowel",
                             "cant poop", "hard stool"],
            "abdominal pain": ["abdominal", "stomach", "belly", "cramp", "cramping",
                               "tummy", "colic", "side pain", "flank"],
            "urinary": ["urinate", "urination", "urine", "pee pee", "bladder", "kidney",
                        "burning when urinating", "frequent urination", "peeing blood"],
            "ear": ["ear", "ear pain", "earache", "ear infection", "hearing", "lost hearing",
                    "deaf", "tinnitus", "ringing", "earwax", "draining from my ear"],
            "eye": ["eye", "eyes", "vision", "blurry", "blurred", "double vision", "sight",
                    "eyelid", "conjunctivitis", "pink eye", "lost vision"],
            "throat": ["throat", "tonsil", "tonsillitis", "sore throat", "hoarse", "voice",
                       "laryngitis"],
            "difficulty swallowing": ["cant swallow", "difficulty swallowing", "trouble swallowing",
                                      "hard to swallow", "painful swallowing", "cant drink",
                                      "food sticking", "drooling", "dysphagia"],
            "bleeding": ["bleeding", "blood", "hemorrhage", "bleed", "bloody", "nosebleed",
                         "spotting", "vomit blood", "blood in"],
            "wound": ["wound", "cut", "laceration", "surgery", "surgical", "incision", "stitch",
                      "open sore", "ulcer", "gash", "tear", "insect bite"],
            "burn injury": ["burned", "burnt", "sunburn", "scald", "chemical burn", "burn on",
                            "burned by", "electric shock", "electrocuted"],
            "trauma": ["trauma", "fall", "fell", "dropped", "fracture", "broken",
                       "broke", "accident", "crash", "hit my head", "hit on the head",
                       "fell off", "car accident", "motorcycle accident", "bruise"],
            "infection": ["infection", "infected", "pus", "purulent", "oozing", "septic",
                          "abscess", "cellulitis", "foul smell", "green discharge", "inflamed"],
            "allergic reaction": ["allergic", "allergy", "allergies", "bee sting", "sting",
                                  "stung", "jellyfish sting", "reaction to", "antihistamine"],
            "anaphylaxis": ["throat closing", "throat is closing", "throat closes", "airway",
                            "angioedema", "anaphylaxis", "anaphylactic", "tongue swelling",
                            "lips swelling", "face is swelling", "cant breathe after"],
            "severe": ["severe", "extreme", "worst", "unbearable", "excruciating", "intense",
                       "agonizing"],
            "chronic": ["chronic", "ongoing", "persistent", "long term", "long standing",
                        "recurring", "keeps coming back", "months now", "years now"],
            "dizziness": ["dizzy", "dizziness", "lightheaded", "light headed", "vertigo",
                          "spinning", "off balance", "unsteady"],
            "loss of consciousness": ["passed out", "pass out", "faint", "blacked out",
                                      "black out", "lost consciousness", "unconscious",
                                      "knocked out", "near syncope", "presyncope"],
            "confusion": ["confused", "confusion", "disoriented", "not making sense",
                          "cant think straight", "memory loss", "dementia"],
            "stroke": ["stroke", "facial droop", "face droop", "droop", "slurred speech",
                       "cant speak", "can not speak", "weak on one side", "numb on one side",
                       "paralysis", "face is sagging"],
            "seizure": ["seizure", "convulsion", "convulsing", "shaking uncontrollably",
                        "epilepsy", "epileptic", "fitting", "went limp"],
            "numbness": ["numb", "numbness", "numbed", "tingle", "tingling", "pins and needles",
                         "lost feeling", "gone flat"],
            "fatigue": ["fatigue", "tired", "exhausted", "exhaustion", "wiped out", "low energy",
                        "weakness", "weak", "no energy", "worn out"],
            "dehydration": ["thirsty", "dehydration", "dehydrated", "dry mouth", "mouth is dry",
                            "dark urine", "not peeing", "no urine", "sunken eyes"],
            "appetite": ["appetite", "weight loss", "lost weight", "not eating", "cant eat",
                         "skipping meals", "eating less", "gain weight"],
            "joint pain": ["joint", "arthritis", "stiff", "stiffness", "knee", "elbow", "hip pain",
                           "shoulder pain", "ankle", "wrist", "back pain", "lower back", "spine"],
            "muscle pain": ["muscle", "muscular", "cricked", "pulled muscle", "crick in my neck"],
            "anxiety": ["anxious", "anxiety", "panic", "restless", "on edge", "cant sleep",
                        "cant function", "jumpy"],
            "depression": ["depressed", "depression", "hopeless", "sad", "numb feelings",
                           "worthless", "no motivation"],
            "mental health crisis": ["kill myself", "suicide", "suicidal", "end my life",
                                     "self harm", "hurt myself", "harm myself", "want to die",
                                     "cant go on", "cut myself", "overdose"],
            "pregnancy": ["pregnant", "pregnancy", "miscarriage", "labor pains", "water broke",
                          "contractions", "ectopic", "hyperemesis", "emesis gravidarum"],
            "diabetes": ["diabetes", "diabetic", "blood sugar", "insulin", "hypoglycemia",
                         "hyperglycemia", "a1c"],
        }
        
        # Compiled once: the scan runs on every intake request and the
        # vocabulary is static.
        self._keyword_patterns = {
            category: [self._keyword_pattern(keyword) for keyword in keywords]
            for category, keywords in self.clinical_keywords.items()
        }
        
        self.acuity_triggers = [
            "severe", "extreme", "worst", "unbearable", "emergency",
            "difficulty breathing", "chest pain", "bleeding", "unconscious",
            # Only language that is unambiguously critical: this list feeds
            # _classify_severity_heuristic, whose "severe" result can push a row
            # to Urgent/Emergency Risk in _determine_acuity.
            "cant breathe", "can not breathe", "cant swallow", "passed out",
            "throat closing", "kill myself", "suicidal", "having a stroke",
            "poison", "overdose", "seizure", "anaphyla", "crushing chest pain",
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
    
    @staticmethod
    def _keyword_pattern(keyword):
        """Word-anchored with inflection: "swell" hits "swelling" and "swells",
        but "hot" cannot hit "photo" or "hotel" and "numb" cannot hit "number".
        The scan runs on the text process_symptoms already lowercased and
        stripped of punctuation, so a keyword containing an apostrophe or hyphen
        can never match - list the stripped form ("cant swallow") instead."""
        return re.compile(r"\b" + re.escape(keyword.lower()) + r"(?:s|es|ed|ing|d)?\b")

    def _extract_symptoms(self, text: str) -> list:
        """Extracts symptoms using the keyword database."""
        symptoms = []

        for symptom_category, patterns in self._keyword_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
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
