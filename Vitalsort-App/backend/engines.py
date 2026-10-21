# backend/engines.py

SYMPTOM_WEIGHTS = {
    "inflammation": 3,
    "lesions": 5,
    "difficulty breathing": 10,
    "chest pain": 10,
    "headache": 1,
    "fever": 4,
    "nausea": 2,
    "rash": 2
}

def calculate_triage_level(symptoms_list):
    """
    Calculates total score based on weights and returns (label, score).
    """
    # Sum the weights, defaulting to 1 for unknown symptoms
    score = sum(SYMPTOM_WEIGHTS.get(symptom.lower(), 1) for symptom in symptoms_list)
    
    if score >= 8:
        return "Level 1: Emergency", score
    elif score >= 5:
        return "Level 2: Urgent", score
    else:
        return "Level 4/5: Routine", score