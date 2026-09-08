from collections import Counter

from core.intent_recognizer import IntentCategory, _TEMPLATES
from evaluation.evaluator import DEFAULT_INTENT_CASES


def test_smoke_dataset_covers_every_intent_evenly():
    counts = Counter(case.expected_intent for case in DEFAULT_INTENT_CASES)

    assert len(DEFAULT_INTENT_CASES) == 57
    assert set(counts) == {intent.value for intent in IntentCategory}
    assert set(counts.values()) == {3}


def test_smoke_messages_are_not_few_shot_templates():
    templates = {text for values in _TEMPLATES.values() for text in values}
    messages = {case.message for case in DEFAULT_INTENT_CASES}

    assert len(messages) == len(DEFAULT_INTENT_CASES)
    assert messages.isdisjoint(templates)
