"""Regression tests for the triage scoring path.

Root cause being guarded: engines.SYMPTOM_WEIGHTS was keyed on clinical
phrases ("chest pain", "difficulty breathing") while the intake engines emit
single-word category keys ("chest", "breathing"). Every lookup missed and fell
through to the default weight of 1, so high-acuity presentations scored as
Routine.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines import SYMPTOM_WEIGHTS, calculate_triage_level
from nlp_engine_ai import AdvancedNLPEngine
from triage_logic import CriticalHeuristicTriage

# Tokens the vision engine can emit from _map_to_clinical_symptoms, kept in sync
# with vision_engine_ai.py.
VISION_SYMPTOM_TOKENS = [
    "inflammation", "erythema", "rash", "lesion", "discoloration",
]

FILLER_TOKENS = [
    "None detected", "No abnormalities detected", "Image analysis pending",
]


@pytest.fixture(scope="session")
def nlp():
    return AdvancedNLPEngine(use_gpt=False)


def test_every_emittable_symptom_has_an_explicit_weight(nlp):
    """The drift guard: any new engine vocabulary must be weighted, not defaulted."""
    nlp_categories = list(nlp.clinical_keywords.keys())

    missing = [
        symptom for symptom in (nlp_categories + VISION_SYMPTOM_TOKENS)
        if symptom not in SYMPTOM_WEIGHTS
    ]
    assert not missing, f"symptoms the engines can emit but scoring ignores: {missing}"


def test_filler_tokens_contribute_no_score():
    for token in FILLER_TOKENS:
        assert calculate_triage_level([token])[1] == 0
    assert calculate_triage_level(FILLER_TOKENS)[0] == "Level 4/5: Routine"


def test_chest_pain_with_breathing_is_not_routine():
    label, score = calculate_triage_level(["pain", "chest", "severe"])

    assert score >= 8
    assert label == "Level 1: Emergency"


def test_isolated_chest_pain_escalates():
    label, score = calculate_triage_level(["pain", "chest"])

    assert label == "Level 1: Emergency"
    assert score >= 8


def test_respiratory_only_presentation_escalates():
    label, _ = calculate_triage_level(["breathing"])

    assert label == "Level 1: Emergency"


def test_low_acuity_single_symptom_stays_routine():
    label, score = calculate_triage_level(["fever"])

    assert score < 5
    assert label == "Level 4/5: Routine"


def test_real_chest_pain_text_is_not_triaged_routine(nlp):
    """End-to-end: raw patient text must not come out the far end as Routine."""
    result = nlp.process_symptoms("I have severe chest pain and difficulty breathing")
    symptoms = result["extracted_symptoms"]

    label, score = calculate_triage_level(symptoms)
    red_flags = CriticalHeuristicTriage.detect_red_flags(
        symptoms, result["inferred_acuity"], "I have severe chest pain and difficulty breathing"
    )

    assert "chest" in symptoms and "breathing" in symptoms
    assert label != "Level 4/5: Routine", f"score={score} symptoms={symptoms}"
    assert red_flags["is_emergency"] is True


def test_severity_modifier_does_not_outweigh_anatomy():
    """'severe'/'chronic' ride along with a real symptom; they must not be the
    biggest term in the table or every complaint becomes an emergency."""
    assert SYMPTOM_WEIGHTS["severe"] < SYMPTOM_WEIGHTS["chest"]
    assert SYMPTOM_WEIGHTS["chronic"] < SYMPTOM_WEIGHTS["fever"]
