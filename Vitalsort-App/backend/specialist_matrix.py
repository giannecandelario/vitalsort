class SpecialistMatrix:
    """
    Pillar 3: Clinical Referral and Urgency Stratification Protocol.
    
    Maps primary symptoms/diagnoses to appropriate medical sub-specialties.
    Helps clinicians route patients to the correct department.
    """
    
    # Maps symptoms/keywords to recommended specialists
    SYMPTOM_TO_SPECIALIST = {
        # Cardiovascular symptoms
        "chest pain": "Cardiology",
        "chest pressure": "Cardiology",
        "palpitations": "Cardiology",
        "syncope": "Cardiology",
        "arrhythmia": "Cardiology",
        "hypertension": "Cardiology",
        
        # Respiratory symptoms
        "difficulty breathing": "Pulmonology",
        "shortness of breath": "Pulmonology",
        "cough": "Pulmonology",
        "asthma": "Pulmonology",
        "wheezing": "Pulmonology",
        
        # Neurological symptoms
        "headache": "Neurology",
        "seizure": "Neurology",
        "dizziness": "Neurology",
        "stroke": "Neurology",
        "paralysis": "Neurology",
        "loss of consciousness": "Neurology",
        
        # Gastrointestinal symptoms
        "nausea": "Gastroenterology",
        "vomiting": "Gastroenterology",
        "abdominal pain": "Gastroenterology",
        "diarrhea": "Gastroenterology",
        "constipation": "Gastroenterology",
        
        # Dermatological symptoms
        "rash": "Dermatology",
        "lesion": "Dermatology",
        "skin infection": "Dermatology",
        "itching": "Dermatology",
        "eczema": "Dermatology",
        
        # Musculoskeletal symptoms
        "joint pain": "Rheumatology",
        "swelling": "Rheumatology",
        "arthritis": "Rheumatology",
        "muscle pain": "Rheumatology",
        
        # Infectious disease
        "fever": "Infectious Disease",
        "infection": "Infectious Disease",
        "sepsis": "Infectious Disease",
        
        # Endocrinology
        "diabetes": "Endocrinology",
        "thyroid": "Endocrinology",
        
        # Psychiatry/Mental Health
        "depression": "Psychiatry",
        "anxiety": "Psychiatry",
        "mental health": "Psychiatry",
        
        # Emergency Medicine (high acuity)
        "severe": "Emergency Medicine",
        "trauma": "Trauma Surgery",
        "bleeding": "Emergency Medicine",
    }
    
    # Specialist contact info & departments
    SPECIALIST_INFO = {
        "Cardiology": {
            "department": "Cardiology Department",
            "description": "Heart and cardiovascular system specialist",
            "typical_wait_time": "24-48 hours",
            "emergency_referral": False
        },
        "Pulmonology": {
            "department": "Pulmonology Department",
            "description": "Lung and respiratory system specialist",
            "typical_wait_time": "24-72 hours",
            "emergency_referral": False
        },
        "Neurology": {
            "department": "Neurology Department",
            "description": "Brain and nervous system specialist",
            "typical_wait_time": "48-72 hours",
            "emergency_referral": True
        },
        "Gastroenterology": {
            "department": "Gastroenterology Department",
            "description": "Digestive system specialist",
            "typical_wait_time": "3-7 days",
            "emergency_referral": False
        },
        "Dermatology": {
            "department": "Dermatology Department",
            "description": "Skin specialist",
            "typical_wait_time": "1-2 weeks",
            "emergency_referral": False
        },
        "Rheumatology": {
            "department": "Rheumatology Department",
            "description": "Joint and autoimmune disease specialist",
            "typical_wait_time": "2-4 weeks",
            "emergency_referral": False
        },
        "Infectious Disease": {
            "department": "Infectious Disease Department",
            "description": "Infection and disease specialist",
            "typical_wait_time": "24-48 hours",
            "emergency_referral": True
        },
        "Endocrinology": {
            "department": "Endocrinology Department",
            "description": "Hormone and metabolic disorder specialist",
            "typical_wait_time": "2-4 weeks",
            "emergency_referral": False
        },
        "Psychiatry": {
            "department": "Psychiatry Department",
            "description": "Mental health specialist",
            "typical_wait_time": "3-7 days",
            "emergency_referral": False
        },
        "Emergency Medicine": {
            "department": "Emergency Department",
            "description": "Acute and emergency care",
            "typical_wait_time": "Immediate",
            "emergency_referral": True
        },
        "Trauma Surgery": {
            "department": "Trauma Center",
            "description": "Surgical trauma specialist",
            "typical_wait_time": "Immediate",
            "emergency_referral": True
        }
    }
    
    @staticmethod
    def get_specialist_recommendation(symptoms: list, acuity_level: str) -> dict:
        """
        Recommends appropriate specialist(s) based on symptoms and acuity.
        
        Args:
            symptoms: List of extracted symptoms
            acuity_level: Acuity classification (Routine, Urgent, Emergency)
            
        Returns:
            Dict with specialist recommendation and details
        """
        # Normalize symptoms
        symptoms_text = " ".join([str(s).lower() for s in symptoms])
        
        # Find all matching specialists
        matched_specialists = set()
        for symptom_keyword, specialist in SpecialistMatrix.SYMPTOM_TO_SPECIALIST.items():
            if symptom_keyword in symptoms_text:
                matched_specialists.add(specialist)
        
        # If no match, default to primary care
        if not matched_specialists:
            matched_specialists.add("Primary Care")
        
        # Get primary specialist (first match)
        primary_specialist = list(matched_specialists)[0] if matched_specialists else "Primary Care"
        
        # Get specialist info
        specialist_info = SpecialistMatrix.SPECIALIST_INFO.get(
            primary_specialist,
            {
                "department": "Primary Care Department",
                "description": "General health assessment",
                "typical_wait_time": "Variable",
                "emergency_referral": False
            }
        )
        
        return {
            "primary_specialist": primary_specialist,
            "recommended_specialists": list(matched_specialists),
            "department": specialist_info.get("department"),
            "description": specialist_info.get("description"),
            "typical_wait_time": specialist_info.get("typical_wait_time"),
            "is_emergency_referral": specialist_info.get("emergency_referral") or acuity_level == "Emergency"
        }
    
    @staticmethod
    def get_all_specialists() -> list:
        """Returns list of all available specialists."""
        return list(SpecialistMatrix.SPECIALIST_INFO.keys())
    
    @staticmethod
    def format_specialist_recommendation(recommendation: dict) -> str:
        """Formats specialist recommendation as readable text."""
        specialist = recommendation.get("primary_specialist", "Unknown")
        department = recommendation.get("department", "")
        wait_time = recommendation.get("typical_wait_time", "Variable")
        
        formatted = f"Recommended Specialist: {specialist} ({department})\nTypical Wait Time: {wait_time}"
        
        if recommendation.get("is_emergency_referral"):
            formatted = "🚨 EMERGENCY REFERRAL - " + formatted
        
        return formatted