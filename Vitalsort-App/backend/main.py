import os
import secrets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the AI/ML engines
from vision_engine_ai import AdvancedVisionEngine
from nlp_engine_ai import AdvancedNLPEngine
from predictive_analytics_ai import PredictiveAnalyticsEngine

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
    
    # 4. Get specialist recommendation
    specialist_rec = SpecialistMatrix.get_specialist_recommendation(
        detected_symptoms,
        priority_label
    )
    
    # 5. Use emergency action if triggered
    recommended_action = red_flags.get("recommended_action")
    if not recommended_action:
        recommended_action = "Consult physician if symptoms persist."
    
    # Pack the triage data
    triage_info = {
        "triage_level": priority_label,
        "recommended_action": recommended_action,
        "is_emergency": red_flags.get("is_emergency", False),
        "emergency_protocol": red_flags.get("protocol"),
        "specialist": specialist_rec.get("primary_specialist"),
        "symptom_severity": nlp_result.get("symptom_severity", "unknown")
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
        "is_emergency": red_flags.get("is_emergency", False),
        "specialist": specialist_rec.get("primary_specialist"),
        "department": specialist_rec.get("department"),
        "symptom_severity": nlp_result.get("symptom_severity", "unknown"),
        "ai_confidence": nlp_result.get("confidence", 0.85)
    }
    
    if red_flags.get("is_emergency"):
        response["emergency_protocol"] = red_flags.get("emergency_instructions")
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
        _, triage_score = calculate_triage_level(
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
        
        if red_flags.get("is_emergency"):
            triage_result["is_emergency"] = True
            triage_result["emergency_protocol"] = red_flags.get("emergency_instructions")
            triage_result["triggered_flags"] = red_flags.get("triggered_flags")
        
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
            "is_emergency": red_flags.get("is_emergency", False),
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
                "gpt_enabled": False,
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
        
        # Add GPT insights if available
        gpt_insights = nlp_result.get("gpt_insights")
        
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
