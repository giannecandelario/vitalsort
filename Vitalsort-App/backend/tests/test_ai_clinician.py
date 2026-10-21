"""Unit guards for the GPT clinician transport, schema guard and escalation rule.

Nothing here touches the network: _request is always stubbed. The behaviours
under test are the ones that keep a model response from silently becoming a
triage decision - refusing to run without a key, rejecting output that does not
match the schema, and refusing to lower an acuity level.
"""
import json
import sys
import urllib.error
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import ai_clinician  # noqa: E402


def analysis(urgency="emergency", citations=None, actions=None):
    return {
        "urgency": urgency,
        "red_flags": ["airway compromise"],
        "differential": [{"condition": "angioedema", "likelihood": "high", "rationale": "swelling"}],
        "immediate_actions": actions or ["Call 911"],
        "specialist": "Emergency Medicine",
        "citations": citations if citations is not None else [
            {"source": "MedlinePlus", "claim": "Swelling with trouble swallowing needs emergency care."}
        ],
        "confidence_note": "Text-only assessment.",
        "model": "gpt-4.1-mini",
    }


def api_response(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-not-real")


def test_no_api_key_means_no_call_and_no_analysis(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args):
        raise AssertionError("network call attempted without a key")

    monkeypatch.setattr(ai_clinician, "_request", forbidden)
    assert ai_clinician.is_configured() is False
    assert ai_clinician.analyze_symptoms("my chest hurts") is None


def test_blank_input_is_not_sent(monkeypatch):
    monkeypatch.setattr(ai_clinician, "_request",
                        lambda *a: (_ for _ in ()).throw(AssertionError("empty text sent")))
    assert ai_clinician.analyze_symptoms("   ") is None


def test_valid_payload_is_parsed(monkeypatch):
    monkeypatch.setattr(ai_clinician, "_request", lambda *a: api_response(analysis()))
    result = ai_clinician.analyze_symptoms("my eyes are swollen and I can't drink water")

    assert result["urgency"] == "emergency"
    assert result["differential"][0]["condition"] == "angioedema"
    assert result["citations"][0]["source"] == "MedlinePlus"


def test_input_is_truncated_before_leaving_the_process(monkeypatch):
    captured = {}

    def spy(prompt, key):
        captured["prompt"] = prompt
        return api_response(analysis(urgency="routine"))

    monkeypatch.setattr(ai_clinician, "_request", spy)
    ai_clinician.analyze_symptoms("pain " * 5000)

    assert len(captured["prompt"]) < ai_clinician.MAX_INPUT_CHARS + 600


@pytest.mark.parametrize("payload", [
    pytest.param("not an object", id="scalar"),
    pytest.param({"urgency": "code_blue"}, id="urgency_not_in_enum"),
    pytest.param({"urgency": "emergency"}, id="missing_fields"),
    pytest.param({**analysis(), "red_flags": "airway"}, id="flags_not_a_list"),
    pytest.param({**analysis(), "differential": [{"condition": "x", "likelihood": "certain"}]},
                 id="likelihood_not_in_enum"),
    pytest.param({**analysis(), "citations": [{"source": "MedlinePlus"}]}, id="citation_missing_claim"),
])
def test_rejects(payload):
    assert ai_clinician._validate(payload) is None


def test_unknown_extra_keys_do_not_disable_the_feature():
    assert ai_clinician._validate({**analysis(), "reasoning": "extra"}) is not None


def test_control_characters_are_stripped_from_model_text():
    cleaned = ai_clinician._clean("angioed\u001b[31ma\x00  swelling\nline")
    assert cleaned == "angioed [31ma swelling line"


@pytest.mark.parametrize("error", [
    pytest.param(urllib.error.HTTPError("u", 429, "rate", {}, None), id="http_429"),
    pytest.param(urllib.error.URLError("dns"), id="dns_failure"),
    pytest.param(TimeoutError(), id="timeout"),
])
def test_api_failure_is_not_fatal(monkeypatch, error):
    def raise_error(*args):
        raise error

    monkeypatch.setattr(ai_clinician, "_request", raise_error)
    assert ai_clinician.analyze_symptoms("headache") is None


def test_unparseable_api_body_is_not_fatal(monkeypatch):
    monkeypatch.setattr(ai_clinician, "_request", lambda *a: {"choices": [{"message": {"content": "{"}}]})
    assert ai_clinician.analyze_symptoms("headache") is None


def test_key_reaches_only_the_authorization_header(monkeypatch):
    seen = {}

    class FakeResponse:
        def read(self):
            return json.dumps(api_response(analysis(urgency="routine"))).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen["headers"] = dict(request.header_items())
        seen["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ai_clinician.analyze_symptoms("sore throat")

    assert seen["headers"]["Authorization"] == "Bearer sk-test-key-not-real"
    assert seen["url"] == ai_clinician.API_URL
    assert result is not None
    assert "sk-test" not in json.dumps(result)


def test_routine_can_be_escalated_to_emergency():
    label, action, escalated = ai_clinician.escalate("Level 4/5: Routine", analysis())

    assert (label, escalated) == ("Level 1: Emergency", True)
    assert "Call 911" in action


def test_urgent_sits_between_routine_and_emergency():
    label, _, escalated = ai_clinician.escalate(
        "Level 4/5: Routine", analysis("urgent", actions=["Be seen today"])
    )
    assert (label, escalated) == ("Level 2: Urgent", True)


def test_a_lower_ai_urgency_never_de_escalates():
    for stored in ("Level 1: Emergency", "Level 2: Urgent", "Level 2: Emergent"):
        label, action, escalated = ai_clinician.escalate(stored, analysis("routine"))
        assert (label, action, escalated) == (stored, None, False)


def test_emergent_and_urgent_spellings_are_the_same_rank():
    assert ai_clinician.label_rank("Level 2: Emergent") == ai_clinician.label_rank("Level 2: Urgent")
    assert ai_clinician.label_rank("Level 1: Emergency") > ai_clinician.label_rank("Level 4/5: Routine")


def test_uncited_emergency_cannot_change_a_stored_level():
    uncited = analysis("emergency", citations=[])
    label, _, escalated = ai_clinician.escalate("Level 4/5: Routine", uncited)

    assert (label, escalated) == ("Level 4/5: Routine", False)


def test_no_analysis_leaves_the_heuristic_level_alone():
    assert ai_clinician.escalate("Level 4/5: Routine", None) == ("Level 4/5: Routine", None, False)
