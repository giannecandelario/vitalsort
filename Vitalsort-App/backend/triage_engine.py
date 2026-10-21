class TriageEngine:
    def __init__(self):
        """
        Initializes the MVP Triage Engine.
        Evaluates extracted NLP clinical context to assign a standard priority level.
        """
        pass

    def calculate_priority(self, clinical_context: dict) -> dict:
        """
        Computes a baseline triage score based on high-acuity triggers 
        and symptom volume.
        """
        acuity = clinical_context.get("inferred_acuity", "Routine")
        symptoms = clinical_context.get("extracted_symptoms", [])
        
        # Strip out the default MVP "None detected" filler to accurately count
        valid_symptoms = [s for s in symptoms if s != "None detected"]

        # Level 2: Emergent (High Acuity Triggered)
        if acuity == "Urgent/Emergency Risk":
            level = "Level 2: Emergent"
            action = "Immediate medical evaluation recommended."
        
        # Level 3: Urgent (Multiple symptoms present)
        elif len(valid_symptoms) >= 2:
            level = "Level 3: Urgent"
            action = "Schedule a clinical visit within 24 hours."
        
        # Level 4/5: Routine (Single symptom or none)
        else:
            level = "Level 4/5: Routine"
            action = "Standard monitoring. Consult a physician if symptoms persist or worsen."

        return {
            "triage_level": level,
            "recommended_action": action,
            "symptom_load": len(valid_symptoms)
        }