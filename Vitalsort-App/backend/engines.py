# backend/engines.py

# Keys must match the vocabulary the intake engines actually emit: the category
# keys of nlp_engine_ai.AdvancedNLPEngine.clinical_keywords and the tokens
# produced by vision_engine_ai._map_to_clinical_symptoms. Phrase keys are kept
# for callers that pass free-text symptoms.
#
# Weights for categories the original table never covered (bleeding, pain,
# swelling, erythema, discoloration) are placed on the existing scale, not
# derived from clinical literature -- review them with a clinician.
#
# Score is additive and the Emergency threshold is 8, so a category that only
# says WHERE the complaint is (headache, joint pain, ear, throat) is weighted 0:
# the generic symptom that fired alongside it already carries the acuity. Without
# this, "my ankle is swollen and it hurts" sums pain + swelling + joint pain past
# 8 and a sprain is stored as Level 1: Emergency.
SYMPTOM_WEIGHTS = {
    # Airway / circulation indicators
    "chest": 10,
    "chest pain": 10,
    "breathing": 10,
    "difficulty breathing": 10,
    "bleeding": 8,
    "anaphylaxis": 10,
    "difficulty swallowing": 8,
    "stroke": 10,
    "seizure": 10,
    "loss of consciousness": 10,
    "mental health crisis": 10,
    "thunderclap headache": 8,
    "confusion": 6,

    # Acute tissue findings
    "lesion": 5,
    "lesions": 5,
    "fever": 4,
    "trauma": 5,
    "wound": 3,
    "burn injury": 3,
    "infection": 4,
    "dehydration": 3,

    # Moderate findings
    "pain": 3,
    "swelling": 3,
    "inflammation": 3,
    "erythema": 3,
    "nausea": 2,
    "rash": 2,
    "discoloration": 2,
    "palpitations": 4,
    "allergic reaction": 4,
    "abdominal pain": 2,
    "diarrhea": 1,
    "constipation": 1,
    "urinary": 3,
    "numbness": 1,
    "dizziness": 1,
    "pregnancy": 3,

    # Low-acuity findings and context. Qualifiers accompany a real symptom, so
    # they stay lighter than the anatomy weights they ride with.
    "severe": 2,
    "chronic": 1,
    "headache": 0,
    "cough": 1,
    "sore throat": 1,
    "throat": 0,
    "ear": 0,
    "eye": 0,
    "joint pain": 0,
    "muscle pain": 0,
    "fatigue": 1,
    "appetite": 1,
    "anxiety": 2,
    "depression": 2,
    "diabetes": 1,
}

# Engine placeholders that carry no clinical signal and must not score.
NON_SCORING_TOKENS = {
    "none detected",
    "no abnormalities detected",
    "image analysis pending",
}

DEFAULT_UNKNOWN_WEIGHT = 1

def calculate_triage_level(symptoms_list):
    """
    Calculates total score based on weights and returns (label, score).
    """
    score = 0
    for symptom in symptoms_list:
        key = symptom.strip().lower()
        if key in NON_SCORING_TOKENS:
            continue
        score += SYMPTOM_WEIGHTS.get(key, DEFAULT_UNKNOWN_WEIGHT)

    if score >= 8:
        return "Level 1: Emergency", score
    elif score >= 5:
        return "Level 2: Urgent", score
    else:
        return "Level 4/5: Routine", score
