"""Intake extraction recall and precision.

The extractor in nlp_engine_ai is a keyword scan, so anything the patient phrases
in an unlisted inflection is silently dropped as "None detected" - and nothing
extracted means nothing scored, which is one layer upstream of the weighting fix.
These cases are the phrasings that produced a Routine row with an empty symptom
list, plus the false positives that a naive substring scan would create.

The transformer pipeline is stubbed: extraction and scoring are deterministic and
keyword-driven, so loading BART would only slow the suite down.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nlp_engine_ai  # noqa: E402
from engines import SYMPTOM_WEIGHTS, calculate_triage_level  # noqa: E402
from triage_logic import CriticalHeuristicTriage  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    """Module scope with a hand-rolled patch: monkeypatch is function scoped and
    cannot be requested by a wider fixture, and loading bart-large-mnli here would
    only slow the suite down - extraction and scoring are keyword driven."""
    original = nlp_engine_ai.pipeline
    nlp_engine_ai.pipeline = lambda *a, **k: None
    try:
        yield nlp_engine_ai.AdvancedNLPEngine()
    finally:
        nlp_engine_ai.pipeline = original


def extract(engine, text):
    """Mirrors process_symptoms' normalisation so cases stay readable."""
    return engine._extract_symptoms(re.sub(r"[^\w\s]", "", text.lower()))


def triage(engine, text):
    symptoms = extract(engine, text)
    label, score = calculate_triage_level(symptoms)
    red_flags = CriticalHeuristicTriage.detect_red_flags(symptoms, "Routine", text)
    if red_flags.get("is_emergency"):
        label = "Level 1: Emergency"
    return symptoms, label, score


# Every category the extractor can emit has to be weighted explicitly, or the
# drift guard in test_triage_scoring.py would already have caught it. This is the
# other half: the vocabulary must not be silently incomplete.
def test_no_category_is_left_unweighted(engine):
    unweighted = [c for c in engine.clinical_keywords if c not in SYMPTOM_WEIGHTS]
    assert not unweighted, f"categories that score nothing: {unweighted}"


@pytest.mark.parametrize("text, expected", [
    pytest.param("My eyes are swollen and I can't drink water", "swelling", id="swollen-inflection"),
    pytest.param("My eyes are swollen and I can't drink water", "difficulty swallowing", id="cannot-drink"),
    pytest.param("My head is itchy", "rash", id="itchy-inflection"),
    pytest.param("I feel dizzy when I stand up", "dizziness", id="dizzy"),
    pytest.param("I have a bad headache and I'm vomiting", "nausea", id="vomiting"),
    pytest.param("I have the worst headache of my life", "thunderclap headache", id="worst-headache"),
    pytest.param("There is a cut on my leg and it's bleeding", "wound", id="cut"),
    pytest.param("I can't keep any food down", "nausea", id="keep-food-down"),
    pytest.param("My throat is closing up after a bee sting", "allergic reaction", id="bee-sting"),
    pytest.param("I have a fever and I'm confused", "confusion", id="confused"),
    pytest.param("I fell off a ladder and my leg might be broken", "trauma", id="fall"),
    pytest.param("I think I'm having a stroke, my face is drooping", "stroke", id="facial-droop"),
    pytest.param("I want to kill myself", "mental health crisis", id="suicidal-ideation"),
    pytest.param("I'm 30 weeks pregnant and cramping", "pregnancy", id="pregnancy"),
    pytest.param("It burns when I urinate and my back hurts", "urinary", id="urination"),
    pytest.param("I can barely hear out of my left ear", "ear", id="ear"),
    pytest.param("My vision has gone blurry since this morning", "eye", id="blurry-vision"),
    pytest.param("I have no appetite and I've lost weight", "appetite", id="weight-loss"),
    pytest.param("My skin is dry and flaky", "rash", id="flaky-skin"),
    pytest.param("I've been exhausted and wiped out all week", "fatigue", id="exhausted"),
    pytest.param("I nearly passed out in the queue", "loss of consciousness", id="near-syncope"),
    pytest.param("My hands and feet are numb and tingle", "numbness", id="numbness"),
    pytest.param("The wound is oozing pus and smells", "infection", id="pus"),
    pytest.param("I'm so anxious I can't function", "anxiety", id="anxiety"),
    pytest.param("I'm very thirsty and my mouth is dry", "dehydration", id="dehydration"),
    pytest.param("I cannot stop wheezing", "breathing", id="wheezing"),
    pytest.param("She was diagnosed with epilepsy as a child", "seizure", id="epilepsy"),
])
def test_phrasing_now_reaches_a_category(engine, text, expected):
    assert expected in extract(engine, text), f"'{text}' extracted {extract(engine, text)}"


@pytest.mark.parametrize("text, category", [
    # Word fragments: "photo" contains "hot", "hear" and "barely" contain "ear",
    # and "burn" sits inside "burned". An unanchored substring scan reports
    # complaints the patient never made.
    pytest.param("I took a photo of the rash", "fever", id="photo_is_not_hot"),
    pytest.param("I can barely hear you", "ear", id="ear_fragment"),
    pytest.param("It is not a burn, the skin is just red", "burn injury", id="burn_is_not_a_wound"),
    pytest.param("My skin peeled after the beach", "urinary", id="peeled_is_not_pee"),
    pytest.param("There is nothing wrong with me today", "pain", id="no_symptoms_stays_empty"),
])
def test_word_fragments_are_not_symptoms(engine, text, category):
    assert category not in extract(engine, text)


@pytest.mark.parametrize("word", ["swelling", "swollen", "swells", "swell"])
def test_swelling_inflections_all_map_to_one_category(engine, word):
    assert "swelling" in extract(engine, f"my ankle is {word}")


@pytest.mark.parametrize("word", ["cough", "coughing", "coughs"])
def test_cough_inflections(engine, word):
    assert "cough" in extract(engine, f"I cannot stop {word}")


def test_airway_compromise_text_becomes_an_emergency(engine):
    """The case that was stored as Routine: facial swelling plus dysphagia."""
    symptoms, label, score = triage(engine, "My eyes are swollen and I can't drink water")

    assert label == "Level 1: Emergency"
    assert score >= 8


def test_anaphylaxis_phrasing_escalates_without_the_exact_trigger(engine):
    # "throat closing" is the stored red-flag phrase; "throat is closing up" is
    # not that substring, so escalation has to come from the weighted symptoms.
    symptoms, label, score = triage(engine, "My throat is closing up after a bee sting")

    assert label == "Level 1: Emergency"


def test_stroke_language_escalates(engine):
    _, label, _ = triage(engine, "I think I'm having a stroke, my face is drooping")

    assert label == "Level 1: Emergency"


def test_self_harm_language_escalates(engine):
    _, label, _ = triage(engine, "I want to kill myself")

    assert label == "Level 1: Emergency"


def test_single_benign_complaints_stay_routine(engine):
    for text in ("I have a rash on my arm", "My head is itchy",
                 "I feel dizzy when I stand up", "I have a mild headache"):
        _, label, _ = triage(engine, text)
        assert label == "Level 4/5: Routine", f"'{text}' escalated to {label}"


@pytest.mark.parametrize("text", [
    # These seven were live rows that came out Level 1: Emergency after the
    # vocabulary widened. Score is additive and the Emergency threshold is 8, so
    # a localisation category stacked on the generic symptom it describes
    # (pain + swelling + joint pain) pushes an ordinary complaint over it.
    "I feel hot and my head aches",
    "My ankle is swollen and it hurts",
    "I have a cough and a sore throat",
    "I have a fever, I'm vomiting and I have diarrhea",
    "I have a bad headache and I'm vomiting",
    "I'm 30 weeks pregnant and my legs are swollen",
])
def test_wider_vocabulary_does_not_invent_emergencies(engine, text):
    symptoms, label, score = triage(engine, text)

    assert label != "Level 1: Emergency", f"'{text}' -> {label} score={score} {symptoms}"


def test_chest_pain_path_is_unchanged(engine):
    """Guards the previously fixed behaviour against a reweighted table."""
    symptoms, label, score = triage(engine, "I have severe chest pain and difficulty breathing")

    assert {"chest", "breathing", "pain", "severe"} <= set(symptoms)
    assert label == "Level 1: Emergency"
    assert score >= 20
