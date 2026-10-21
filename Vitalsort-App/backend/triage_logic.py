class CriticalHeuristicTriage:
    """
    Pillar 2: Automated Differential Triage and Inference Engine.
    
    Critical Heuristic Triage (Red Flag Protocols): An un-bypassable heuristic 
    override that detects high-acuity indicators like systemic sepsis markers 
    or acute anaphylactic language. When triggered, initiates immediate systemic 
    halt and displays standardized emergency care protocols.
    """
    
    # Emergency trigger keywords mapped to categories
    EMERGENCY_TRIGGERS = {
        "respiratory": [
            "difficulty breathing", "shortness of breath", "apnea", 
            "gasping", "unable to breathe", "severe dyspnea"
        ],
        "cardiovascular": [
            "chest pain", "chest pressure", "severe palpitations", 
            "syncope", "fainting", "collapse", "acute heart attack"
        ],
        "sepsis": [
            "systemic sepsis", "septic shock", "high fever with chills",
            "fever with confusion", "rapid heartbeat", "signs of infection"
        ],
        "anaphylaxis": [
            "anaphylactic", "severe swelling", "airway compromise", 
            "severe allergic reaction", "throat closing", "angioedema"
        ],
        "trauma": [
            "severe bleeding", "uncontrolled bleeding", "severe trauma",
            "loss of consciousness", "traumatic injury"
        ],
        "neurological": [
            "seizure", "loss of consciousness", "severe headache with fever",
            "stroke symptoms", "sudden paralysis"
        ]
    }
    
    # Standard emergency care protocols by trigger category
    EMERGENCY_PROTOCOLS = {
        "respiratory": "Call 911 immediately. Position patient upright. Prepare for emergency airway management.",
        "cardiovascular": "Call 911 immediately. Begin CPR if unconscious and unresponsive. Administer oxygen if available.",
        "sepsis": "Call 911 immediately. Initiate IV fluids if trained. Monitor vital signs continuously.",
        "anaphylaxis": "Call 911 immediately. Administer epinephrine IM 0.3-0.5mg if available. Position supine.",
        "trauma": "Call 911 immediately. Apply direct pressure to bleeding. Stabilize spine if indicated.",
        "neurological": "Call 911 immediately. Protect airway. Monitor consciousness and vital signs."
    }
    
    @staticmethod
    def detect_red_flags(symptom_list: list, acuity_level: str, raw_text: str = "") -> dict:
        """
        Detects high-acuity indicators and triggers immediate emergency protocol.
        
        Args:
            symptom_list: List of extracted symptoms
            acuity_level: Acuity classification from NLPEngine (e.g., "Urgent/Emergency Risk")
            raw_text: Original user text (for phrase matching)
            
        Returns:
            Dict with is_emergency flag, triggered categories, and protocol instructions
        """
        triggered_flags = []
        triggered_categories = set()
        
        # Normalize symptom text for keyword matching
        symptom_text = " ".join([str(s).lower() for s in symptom_list])
        raw_text_lower = raw_text.lower() if raw_text else ""
        
        # Check each category for emergency keywords
        for category, keywords in CriticalHeuristicTriage.EMERGENCY_TRIGGERS.items():
            # Check extracted symptoms
            if any(keyword in symptom_text for keyword in keywords):
                triggered_flags.append(category)
                triggered_categories.add(category)
            # Also check raw text for phrase matching (e.g., "chest pain")
            elif any(keyword in raw_text_lower for keyword in keywords):
                triggered_flags.append(category)
                triggered_categories.add(category)
        
        # Determine if emergency based on flags or acuity level
        if triggered_flags or acuity_level == "Urgent/Emergency Risk":
            # Select primary protocol (first triggered category, or respiratory as default)
            primary_category = triggered_flags[0] if triggered_flags else "respiratory"
            protocol_text = CriticalHeuristicTriage.EMERGENCY_PROTOCOLS.get(
                primary_category, 
                "Call 911 immediately. Patient requires emergency evaluation."
            )
            
            return {
                "is_emergency": True,
                "triggered_flags": triggered_flags,
                "primary_category": primary_category,
                "protocol": "IMMEDIATE SYSTEMIC HALT - EMERGENCY PROTOCOL ACTIVATED",
                "emergency_instructions": protocol_text,
                "recommended_action": f"🚨 EMERGENCY: {protocol_text}",
                "severity": "CRITICAL"
            }
        
        # Non-emergency response
        return {
            "is_emergency": False,
            "triggered_flags": [],
            "primary_category": None,
            "protocol": None,
            "emergency_instructions": None,
            "recommended_action": None,
            "severity": "NORMAL"
        }
    
    @staticmethod
    def validate_intake_completeness(symptom_data: dict) -> dict:
        """
        Validates that critical intake data is present before proceeding.
        
        Args:
            symptom_data: Dict containing extracted symptom information
            
        Returns:
            Dict with validation status and any missing fields
        """
        required_fields = ["extracted_symptoms", "inferred_acuity"]
        missing_fields = [field for field in required_fields if not symptom_data.get(field)]
        
        return {
            "is_valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "data_quality": "COMPLETE" if len(missing_fields) == 0 else "INCOMPLETE"
        }
    
    @staticmethod
    def classify_urgency_tier(triage_score: int, has_emergency_flags: bool) -> str:
        """
        Maps triage score and emergency status to standardized urgency tier.
        
        Args:
            triage_score: Numeric score from calculate_triage_level
            has_emergency_flags: Boolean indicating emergency flag detection
            
        Returns:
            Urgency tier label (Emergency, Urgent, or Routine)
        """
        if has_emergency_flags or triage_score >= 20:
            return "Level 1: Emergency"
        elif triage_score >= 8:
            return "Level 2: Urgent"
        else:
            return "Level 4/5: Routine"