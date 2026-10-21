import os
import secrets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the engines (relative imports)
from vision_engine import VisionEngine
from nlp_engine import NLPEngine
from triage_engine import TriageEngine
from db_engine import DBEngine
from engines import calculate_triage_level
from triage_logic import CriticalHeuristicTriage
from pdf_exporter import PDFExporter
from specialist_matrix import SpecialistMatrix
from contacts import ContactManager
from analytics_engine import AnalyticsEngine
from fastapi.responses import StreamingResponse

# Instantiate the engines
vision_engine = VisionEngine()
nlp_engine = NLPEngine()
triage_engine = TriageEngine()
db_engine = DBEngine()
contact_manager = ContactManager()
analytics = AnalyticsEngine()

# Initialize the FastAPI application
app = FastAPI(title="Vitalsort MVP Core Architecture")

# Configure CORS for decoupled frontend communication
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
# DATA GOVERNANCE: Zero-Retention Purge Protocol
# ---------------------------------------------------------
def secure_zero_fill_purge(filepath: str):
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        with open(filepath, "ba+", buffering=0) as f:
            f.seek(0)
            f.write(b'\x00' * file_size)
        os.remove(filepath)

# ---------------------------------------------------------
# PILLAR 1: Dynamic Natural Language Intake (REST)
# ---------------------------------------------------------
@app.post("/api/intake/nlp")
async def process_nlp(data: dict):
    raw_text = data.get("text", "")
    
    # 1. Process text with NLP engine
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
    
    # 5. Use emergency action if triggered, otherwise use standard action
    recommended_action = red_flags.get("recommended_action")
    if not recommended_action:
        recommended_action = "Consult physician if symptoms persist."
    
    # Pack the triage data
    triage_info = {
        "triage_level": priority_label,
        "recommended_action": recommended_action,
        "is_emergency": red_flags.get("is_emergency", False),
        "emergency_protocol": red_flags.get("protocol"),
        "specialist": specialist_rec.get("primary_specialist")
    }
    
    # 6. Save the record
    record_id = db_engine.insert_record(
        raw_text=raw_text,
        nlp_data=nlp_result,
        triage_data=triage_info,
        triage_score=score
    )
    
    # Return with emergency flag and specialist info
    response = {
        "status": "success",
        "record_id": record_id,
        "is_emergency": red_flags.get("is_emergency", False),
        "specialist": specialist_rec.get("primary_specialist"),
        "department": specialist_rec.get("department"),
        "recommended_specialists": specialist_rec.get("recommended_specialists")
    }
    
    if red_flags.get("is_emergency"):
        response["emergency_protocol"] = red_flags.get("emergency_instructions")
        response["triggered_flags"] = red_flags.get("triggered_flags")
    
    return response

# ---------------------------------------------------------
# PILLAR 2: Visual Ingestion Interface (WebSocket)
# ---------------------------------------------------------
@app.websocket("/ws/visual-ingestion")
async def visual_ingestion(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the raw binary image chunk
        data = await websocket.receive_bytes()
        
        # 1. Analyze the image
        visual_context = vision_engine.analyze_image(data)
        
        # 2. Calculate triage priority based on visual symptoms
        triage_result = triage_engine.calculate_priority(visual_context)
        
        # 3. Calculate triage score
        _, triage_score = calculate_triage_level(
            visual_context.get("extracted_symptoms", [])
        )
        
        # 4. Check for red flag emergencies
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
        
        # Add emergency info to triage result if triggered
        if red_flags.get("is_emergency"):
            triage_result["is_emergency"] = True
            triage_result["emergency_protocol"] = red_flags.get("emergency_instructions")
            triage_result["triggered_flags"] = red_flags.get("triggered_flags")
        
        # Add specialist info
        triage_result["specialist"] = specialist_rec.get("primary_specialist")
        
        # 6. Save the visual assessment to the queue
        record_id = db_engine.insert_record(
            raw_text="[IMAGE UPLOADED]", 
            nlp_data=visual_context, 
            triage_data=triage_result,
            triage_score=triage_score
        )

        # 7. Confirm success to the frontend
        await websocket.send_json({
            "status": "chunk_received",
            "record_id": record_id,
            "triage": triage_result,
            "is_emergency": red_flags.get("is_emergency", False),
            "specialist": specialist_rec.get("primary_specialist")
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
    return {"status": "online", "architecture": "Vitalsort MVP Core"}

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
    """
    Exports a triage record as a professional PDF clinical dossier.
    """
    try:
        all_records = db_engine.get_active_queue()
        record = None
        for r in all_records:
            if r['id'] == record_id:
                record = r
                break
        
        if not record:
            return {"error": f"Record ID {record_id} not found"}
        
        # Generate PDF
        pdf_buffer = PDFExporter.generate_clinical_dossier(record)
        
        # Return as downloadable file
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

@app.get("/api/contacts/all")
async def get_all_contacts():
    """Retrieves all patient contacts (admin only)."""
    try:
        contacts = contact_manager.get_all_contacts()
        return {
            "status": "success",
            "count": len(contacts),
            "data": contacts
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

@app.get("/api/analytics/trends/{days}")
async def get_trends(days: int = 30):
    """Returns trends for specified number of days."""
    try:
        return {
            "status": "success",
            "distribution": analytics.get_triage_distribution(days),
            "symptoms": analytics.get_symptom_frequency(days),
            "daily_trends": analytics.get_acuity_trends(days)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/analytics/stats")
async def get_system_statistics():
    """Returns overall system statistics."""
    try:
        return {
            "status": "success",
            "statistics": analytics.get_system_stats()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}