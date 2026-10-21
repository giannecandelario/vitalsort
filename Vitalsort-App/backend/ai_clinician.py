"""
AI clinician: GPT-backed analysis of the free-text symptom input.

Transport is stdlib urllib against the OpenAI Chat Completions API so the
feature adds no dependency (blueprint.md, "Native Preference") and leaves the
retired openai==0.27.8 client unused.

Two invariants:
  * The model may ESCALATE the deterministic triage level, never de-escalate it.
  * Nothing here may break intake. No key, an unreachable API or output that
    fails schema validation all return None and the heuristic result stands.
"""

import json
import os
import re
import urllib.error
import urllib.request

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
TIMEOUT_SECONDS = 20
MAX_INPUT_CHARS = 2000

# Rank ladder shared by the heuristic labels ("Level 1: Emergency",
# "Level 2: Urgent"/"Level 2: Emergent", "Level 4/5: Routine") and the model's
# own vocabulary, so escalation can be compared across both.
_LABEL_RANK = {"1": 4, "2": 3, "3": 2, "4": 1, "5": 0}
_URGENCY_RANK = {"emergency": 4, "urgent": 3, "routine": 1}
_URGENCY_LABEL = {4: "Level 1: Emergency", 3: "Level 2: Urgent", 1: "Level 4/5: Routine"}

SYSTEM_PROMPT = """You are a clinical decision support analyst for a triage intake tool.

You are not diagnosing, prescribing, or treating. You analyse the patient's own words.

Rules:
1. Report only findings supported by the text you are given. Do not invent symptoms,
   history, demographics or exposures.
2. Every clinical claim you return must appear in "citations" with an authoritative
   public source (MedlinePlus, NHS, CDC, WHO, Merck Manual professional edition, or an
   AHA/ACC/ACS/NICE guideline). Cite a source only when you are confident the claim is
   actually documented there. If a claim cannot be cited, omit the claim.
3. Never cite a specific document title, URL, page number or year you are not sure of.
4. Escalate rather than reassure when the picture is ambiguous.
5. Return urgency="emergency" when the text suggests a time-critical condition: airway
   compromise, difficulty swallowing or speaking, facial or throat swelling, chest pain
   with breathlessness, stroke signs, uncontrolled bleeding, anaphylaxis, altered
   consciousness, or a suicide/homicide risk statement.
6. Return urgency="urgent" for same-day or within-24-hours evaluation needs, and
   "routine" only when the text supports watchful waiting.
7. "immediate_actions" are safe first-aid and disposition steps, not treatments.
8. Return only the JSON object, with every field present."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "urgency": {"type": "string", "enum": ["routine", "urgent", "emergency"]},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "differential": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "likelihood": {"type": "string", "enum": ["high", "medium", "low"]},
                    "rationale": {"type": "string"},
                },
                "required": ["condition", "likelihood", "rationale"],
                "additionalProperties": False,
            },
        },
        "immediate_actions": {"type": "array", "items": {"type": "string"}},
        "specialist": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["source", "claim"],
                "additionalProperties": False,
            },
        },
        "confidence_note": {"type": "string"},
    },
    "required": [
        "urgency",
        "red_flags",
        "differential",
        "immediate_actions",
        "specialist",
        "citations",
        "confidence_note",
    ],
    "additionalProperties": False,
}


def _load_dotenv():
    """Minimal KEY=VALUE reader for backend/.env so a local key works without
    exporting it in every shell. Real environment variables always win, and the
    values are never logged."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value


_load_dotenv()


def api_key():
    """Read the key from the environment only. Never logged, stored or returned."""
    return os.environ.get("OPENAI_API_KEY", "").strip()


def is_configured():
    return bool(api_key())


def model_name():
    return os.environ.get("VITALSORT_GPT_MODEL", DEFAULT_MODEL)


def label_rank(label: str) -> int:
    match = re.search(r"level\s*(\d)", (label or "").lower())
    return _LABEL_RANK.get(match.group(1), 0) if match else 0


def _clean(value, limit=400):
    if value is None:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _clean_list(value, limit=6):
    if not isinstance(value, list):
        return None
    items = [_clean(item, 240) for item in value if _clean(item, 240)]
    return items[:limit]


def _validate(payload):
    """
    Guard the boundary: model output is untrusted input.

    Missing keys are rejected because a half-answer read as a clinical opinion is
    worse than no answer. Unknown extra keys are ignored so a model update that
    adds a field cannot silently disable the feature.
    """
    if not isinstance(payload, dict):
        return None
    if any(field not in payload for field in RESPONSE_SCHEMA["required"]):
        return None
    urgency = payload.get("urgency")
    if urgency not in _URGENCY_RANK:
        return None

    differential = []
    for entry in payload.get("differential") or []:
        if not isinstance(entry, dict):
            return None
        condition, likelihood, rationale = (
            _clean(entry.get("condition"), 160),
            entry.get("likelihood"),
            _clean(entry.get("rationale"), 240),
        )
        if not condition or likelihood not in ("high", "medium", "low"):
            return None
        differential.append({"condition": condition, "likelihood": likelihood, "rationale": rationale})

    citations = []
    for entry in payload.get("citations") or []:
        if not isinstance(entry, dict):
            return None
        source, claim = _clean(entry.get("source"), 120), _clean(entry.get("claim"), 240)
        if not source or not claim:
            return None
        citations.append({"source": source, "claim": claim})

    red_flags = _clean_list(payload.get("red_flags"))
    actions = _clean_list(payload.get("immediate_actions"))
    if red_flags is None or actions is None:
        return None

    return {
        "urgency": urgency,
        "red_flags": red_flags,
        "differential": differential[:3],
        "immediate_actions": actions,
        "specialist": _clean(payload.get("specialist"), 80),
        "citations": citations[:6],
        "confidence_note": _clean(payload.get("confidence_note"), 240),
        "model": model_name(),
    }


def _request(prompt_text, key):
    body = json.dumps(
        {
            "model": model_name(),
            "temperature": 0.2,
            "max_tokens": 900,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "triage_analysis", "strict": True, "schema": RESPONSE_SCHEMA},
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def analyze_symptoms(raw_text: str, heuristic_symptoms=None, heuristic_level: str = ""):
    """Return a validated analysis dict, or None when it must not be relied on."""
    key = api_key()
    text = (raw_text or "").strip()
    if not key or not text:
        return None

    context = ""
    if heuristic_symptoms or heuristic_level:
        context = (
            "\n\nLocal heuristic result for reference only - do not defer to it:"
            f"\nkeyword-matched categories: {heuristic_symptoms or []}"
            f"\nassigned level: {heuristic_level or 'unset'}"
        )

    prompt = (
        "Analyse this patient-authored symptom description for triage purposes.\n"
        f'"""{text[:MAX_INPUT_CHARS]}"""\n{context}'
    )

    try:
        response = _request(prompt, key)
    except urllib.error.HTTPError as error:
        # The response body can echo the request, so only the status is reported.
        print(f"AI clinician: OpenAI HTTP {error.code}")
        return None
    except Exception as error:
        print(f"AI clinician unavailable: {type(error).__name__}")
        return None

    try:
        content = response["choices"][0]["message"]["content"]
        analysis = _validate(json.loads(content))
    except (KeyError, IndexError, TypeError, ValueError) as error:
        print(f"AI clinician: malformed API response ({type(error).__name__})")
        return None

    if analysis is None:
        print("AI clinician: output failed schema validation, ignoring it")
    return analysis


def escalate(priority_label: str, analysis):
    """
    Apply the analysis to a deterministic level. Returns
    (priority_label, recommended_action, escalated) and never lowers a level.

    An uncited emergency claim cannot escalate: the user's grounding rule is that
    verified data is what earns the authority to change a stored triage decision.
    """
    if not analysis:
        return priority_label, None, False

    ai_rank = _URGENCY_RANK[analysis["urgency"]]
    if ai_rank <= label_rank(priority_label):
        return priority_label, None, False

    if not analysis["citations"]:
        return priority_label, None, False

    label = _URGENCY_LABEL[ai_rank]
    first_action = analysis["immediate_actions"][0] if analysis["immediate_actions"] else ""
    action = f"AI clinician escalated to {label}."
    if first_action:
        action += f" {first_action}"
    return label, action[:400], True
