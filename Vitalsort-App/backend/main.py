import os
import asyncio
import secrets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the AI/ML engines
from vision_engine_ai import AdvancedVisionEngine
from nlp_engine_ai import AdvancedNLPEngine
from predictive_analytics_ai import PredictiveAnalyticsEngine
import ai_clinician

# Import existing engines
from triage_engine import TriageEngine
from db_engine import DBEngine
from engines import calculate_triage_level
from triage_logic import CriticalHeuristicTriage
from pdf_exporter import PDFExporter
from specialist_matrix import SpecialistMatrix
from contacts import ContactManager
from analytics_engine import AnalyticsEngine
from fastapi.responses import StreamingResponse

# Initialize AI/ML engines
vision_engine = AdvancedVisionEngine()
nlp_engine = AdvancedNLPEngine(use_gpt=False)  # Set to True if you have OpenAI API key
predictive_engine = PredictiveAnalyticsEngine()

# Initialize existing engines
triage_engine = TriageEngine()
db_engine = DBEngine()
contact_manager = ContactManager()
analytics = AnalyticsEngine()

# Initialize FastAPI
app = FastAPI(title="Vitalsort AI/ML MVP")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../temp_processing"))
os.makedirs(TEMP_DIR, exist_ok=True)

# Emitted when a row reaches Level 1 without an un-bypassable trigger phrase, so
# its only source is the weighted score or the AI clinician.
DEFAULT_EMERGENCY_INSTRUCTIONS = (
    "Call 911 or go to the nearest emergency department now."
)

# ---------------------------------------------------------
# PILLAR 1: Dynamic Natural Language Intake (REST)
# ---------------------------------------------------------
@app.post("/api/intake/nlp")
async def process_nlp(data: dict):
    raw_text = data.get("text", "")
    
    # 1. Process text with ADVANCED NLP engine
    nlp_result = nlp_engine.process_symptoms(raw_text)
    detected_symptoms = nlp_result.get("extracted_symptoms", [])
    inferred_acuity = nlp_result.get("inferred_acuity", "Routine")
    
    # 2. Calculate triage score
    priority_label, score = calculate_triage_level(detected_symptoms)
    
    # 3. Check for red flag emergencies
    red_flags = CriticalHeuristicTriage.detect_red_flags(detected_symptoms, inferred_acuity, raw_text)
    
    # An un-bypassable red flag outranks the weighted score. Without this the
    # stored row understates acuity even though the client was told it is an
    # emergency, and the queue, analytics and dossier all inherit the error.
    if red_flags.get("is_emergency"):
        priority_label = "Level 1: Emergency"

    # 3b. GPT clinician. Runs in a worker thread because urllib blocks, and may
    # only raise the level - a lower AI urgency is discarded so the deterministic
    # engines stay the floor of every stored row.
    analysis = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: ai_clinician.analyze_symptoms(
            raw_text, detected_symptoms, priority_label
        ),
    )
    priority_label, escalated_action, escalated = ai_clinician.escalate(
        priority_label, analysis
    )
    # Level 1 is Level 1 whichever engine put it there: an un-bypassable trigger
    # phrase, the weighted score, or the AI clinician. A row stored as
    # "Level 1: Emergency" that the client is not told is an emergency is the
    # same incoherence as the old reverse, where every acuity flag was an
    # emergency the score never justified.
    is_emergency = priority_label == "Level 1: Emergency"

    # 4. Get specialist recommendation
    specialist_rec = SpecialistMatrix.get_specialist_recommendation(
        detected_symptoms,
        priority_label
    )
    if is_emergency:
        # A Level 1 row referring to Primary Care or Dermatology would contradict
        # its own acuity, so every emergency routes to the ED.
        specialist_rec = {
            "primary_specialist": "Emergency Medicine",
            "department": "Emergency Department",
        }

    # 5. Use emergency action if triggered
    recommended_action = escalated_action or red_flags.get("recommended_action")
    if is_emergency:
        # A score-only or AI-only Level 1 has no protocol text, and must not fall
        # through to the reassuring wording below.
        recommended_action = recommended_action or f"🚨 EMERGENCY: {DEFAULT_EMERGENCY_INSTRUCTIONS}"
    if not recommended_action:
        recommended_action = "Consult physician if symptoms persist."
    
    # Pack the triage data
    triage_info = {
        "triage_level": priority_label,
        "recommended_action": recommended_action,
        "is_emergency": is_emergency,
        "emergency_protocol": red_flags.get("protocol"),
        "specialist": specialist_rec.get("primary_specialist"),
        "symptom_severity": nlp_result.get("symptom_severity", "unknown"),
        "ai_analysis": analysis
    }
    
    # 6. Save the record
    record_id = db_engine.insert_record(
        raw_text=raw_text,
        nlp_data=nlp_result,
        triage_data=triage_info,
        triage_score=score
    )
    
    # Return response
    response = {
        "status": "success",
        "record_id": record_id,
        "priority_level": priority_label,
        "is_emergency": is_emergency,
        "specialist": specialist_rec.get("primary_specialist"),
        "department": specialist_rec.get("department"),
        "symptom_severity": nlp_result.get("symptom_severity", "unknown"),
        "ai_confidence": nlp_result.get("confidence", 0.85),
        "ai_analysis": analysis,
        "ai_escalated": escalated
    }
    
    if is_emergency:
        # Fixed wording, not model text: an emergency banner must not be
        # phrased by whatever the API returned.
        response["emergency_protocol"] = red_flags.get("emergency_instructions") or (
            DEFAULT_EMERGENCY_INSTRUCTIONS
        )
        response["triggered_flags"] = red_flags.get("triggered_flags")
    
    return response

# ---------------------------------------------------------
# PILLAR 2: Visual Ingestion Interface (WebSocket with AI)
# ---------------------------------------------------------
@app.websocket("/ws/visual-ingestion")
async def visual_ingestion(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_bytes()
        
        # 1. Analyze with ADVANCED VISION ENGINE
        visual_context = vision_engine.analyze_image(data)
        
        # 2. Calculate triage priority
        triage_result = triage_engine.calculate_priority(visual_context)
        
        # 3. Calculate triage score
        weighted_label, triage_score = calculate_triage_level(
            visual_context.get("extracted_symptoms", [])
        )
        
        # 4. Check for red flags
        red_flags = CriticalHeuristicTriage.detect_red_flags(
            visual_context.get("extracted_symptoms", []),
            visual_context.get("inferred_acuity", "Routine"),
            "[IMAGE UPLOADED]"
        )
        
        # 5. Get specialist recommendation
        specialist_rec = SpecialistMatrix.get_specialist_recommendation(
            visual_context.get("extracted_symptoms", []),
            triage_result.get("triage_level", "")
        )
        
        # A picture cannot declare an emergency. Colour dominance and edge density
        # fire on warm-toned and merely noisy images, so Level 1 on this path needs
        # an un-bypassable trigger phrase; a score in the Emergency band from signs
        # alone is stored as Urgent.
        is_emergency = red_flags.get("is_emergency", False)
        if is_emergency:
            triage_result["is_emergency"] = True
            triage_result["triage_level"] = "Level 1: Emergency"
            triage_result["emergency_protocol"] = red_flags.get("emergency_instructions") or (
                DEFAULT_EMERGENCY_INSTRUCTIONS
            )
            triage_result["triggered_flags"] = red_flags.get("triggered_flags")
        elif weighted_label == "Level 1: Emergency":
            triage_result["triage_level"] = "Level 2: Urgent"
            triage_result["recommended_action"] = "Seek clinical evaluation the same day."
        
        triage_result["specialist"] = specialist_rec.get("primary_specialist")
        
        # 6. Save record
        record_id = db_engine.insert_record(
            raw_text="[IMAGE UPLOADED]", 
            nlp_data=visual_context, 
            triage_data=triage_result,
            triage_score=triage_score
        )

        # 7. Send response
        await websocket.send_json({
            "status": "chunk_received",
            "record_id": record_id,
            "triage": triage_result,
            "is_emergency": is_emergency,
            "specialist": specialist_rec.get("primary_specialist"),
            "ai_confidence": visual_context.get("confidence", 0.85)
        })
        
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/health")
async def system_health_check():
    return {"status": "online", "architecture": "Vitalsort AI/ML MVP"}

# ---------------------------------------------------------
# PILLAR 3: Provider Dashboard Data (REST)
# ---------------------------------------------------------
@app.get("/api/queue")
async def get_triage_queue():
    """Returns the active list of triaged patients."""
    queue_data = db_engine.get_active_queue()
    return {"status": "success", "queue_count": len(queue_data), "data": queue_data}

# ---------------------------------------------------------
# PILLAR 4: Standardized Pre-Consultation Reporting Interface
# ---------------------------------------------------------
@app.get("/api/export/dossier/{record_id}")
async def export_clinical_dossier(record_id: int):
    """Exports a triage record as a professional PDF clinical dossier."""
    try:
        all_records = db_engine.get_active_queue()
        record = None
        for r in all_records:
            if r['id'] == record_id:
                record = r
                break
        
        if not record:
            return {"error": f"Record ID {record_id} not found"}
        
        pdf_buffer = PDFExporter.generate_clinical_dossier(record)
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=vitalsort_dossier_{record_id}.pdf"
            }
        )
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# PILLAR 5: Specialist Referral & Classification
# ---------------------------------------------------------
@app.get("/api/specialists/list")
async def get_all_specialists():
    """Returns list of all available specialists."""
    try:
        specialists = SpecialistMatrix.get_all_specialists()
        return {
            "status": "success",
            "count": len(specialists),
            "specialists": specialists
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------
# PILLAR 6: Patient Contact & Follow-up Management
# ---------------------------------------------------------
@app.post("/api/contact/save")
async def save_patient_contact(data: dict):
    """Saves patient contact information for follow-up coordination."""
    try:
        name = data.get("name", "")
        email = data.get("email", "")
        phone = data.get("phone", "")
        notes = data.get("notes", "")
        
        if not name or not email or not phone:
            return {"status": "error", "error": "Missing required fields"}
        
        contact_id = contact_manager.save_contact(name, email, phone, notes)
        
        return {
            "status": "success",
            "contact_id": contact_id,
            "message": "Contact information saved successfully"
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/contacts/pending")
async def get_pending_follow_ups():
    """Retrieves contacts pending follow-up."""
    try:
        pending = contact_manager.get_pending_follow_ups()
        return {
            "status": "success",
            "count": len(pending),
            "data": pending
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------
# PILLAR 7: Analytics & Historical Trends
# ---------------------------------------------------------
@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard():
    """Returns comprehensive analytics dashboard data."""
    try:
        return {
            "status": "success",
            "system_stats": analytics.get_system_stats(),
            "triage_distribution": analytics.get_triage_distribution(30),
            "symptom_frequency": analytics.get_symptom_frequency(30),
            "acuity_trends": analytics.get_acuity_trends(7),
            "specialist_stats": analytics.get_specialist_referral_stats()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------
# NEW: AI/ML FEATURES
# ---------------------------------------------------------

# Feature 16: Real ML Models
@app.get("/api/ai/model-status")
async def get_model_status():
    """Returns status of AI/ML models."""
    return {
        "status": "success",
        "models": {
            "vision": {
                "name": "Advanced Vision Engine",
                "model": "ResNet50 (ImageNet pretrained)",
                "status": "active",
                "confidence": 0.85
            },
            "nlp": {
                "name": "Advanced NLP Engine",
                "model": "BART Large MNLI + BioBERT",
                "status": "active",
                # Reported from reality: the clinician is live only when a key exists.
                "gpt_enabled": ai_clinician.is_configured(),
                "gpt_model": ai_clinician.model_name(),
                "confidence": 0.87
            },
            "predictions": {
                "name": "Predictive Analytics",
                "model": "Random Forest Ensemble",
                "status": "active",
                "confidence": 0.82
            }
        }
    }

# Feature 17: Predictive Analytics
@app.get("/api/predictions/patient/{record_id}")
async def get_patient_predictions(record_id: int):
    """Get outcome predictions for a specific patient."""
    try:
        # Get patient record
        all_records = db_engine.get_active_queue()
        record = None
        for r in all_records:
            if r['id'] == record_id:
                record = r
                break
        
        if not record:
            return {"status": "error", "error": "Record not found"}
        
        # Generate predictions
        predictions = predictive_engine.predict_patient_outcomes(record)
        
        return predictions
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/predictions/population")
async def get_population_predictions():
    """Get predictive insights for entire patient population."""
    try:
        insights = predictive_engine.get_population_insights()
        return insights
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Feature 18: Advanced NLU with Insights
@app.post("/api/ai/nlu-advanced")
async def advanced_nlu(data: dict):
    """Advanced NLU with additional insights."""
    try:
        text = data.get("text", "")
        
        # Process with advanced NLP
        nlp_result = nlp_engine.process_symptoms(text)

        # The retired nlp_engine use_gpt path never ran; this endpoint now calls the
        # same clinician the intake path uses so "insights" is not a permanent null.
        gpt_insights = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: ai_clinician.analyze_symptoms(text),
        )
        
        return {
            "status": "success",
            "symptoms": nlp_result.get("extracted_symptoms"),
            "severity": nlp_result.get("symptom_severity"),
            "acuity": nlp_result.get("inferred_acuity"),
            "confidence": nlp_result.get("confidence"),
            "gpt_insights": gpt_insights
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------
# AI/ML MANAGEMENT
# ---------------------------------------------------------

@app.post("/api/ai/enable-gpt")
async def enable_gpt(data: dict):
    """Enable GPT integration (requires OpenAI API key)."""
    try:
        api_key = data.get("api_key", "")
        
        if not api_key:
            return {"status": "error", "error": "API key required"}
        
        # Reinitialize NLP engine with GPT
        global nlp_engine
        nlp_engine = AdvancedNLPEngine(use_gpt=True, openai_api_key=api_key)
        
        return {
            "status": "success",
            "message": "GPT integration enabled",
            "model": "gpt-3.5-turbo"
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}
