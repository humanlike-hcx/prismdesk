from collections import Counter

from core.intent_recognizer import RouteCategory, _LLM_FEW_SHOTS, _TEMPLATES
from evaluation.evaluator import ROUTE_BLIND_CASES, ROUTE_DEV_CASES, ROUTE_VALIDATION_CASES


def test_each_route_dataset_covers_every_route_evenly():
    for cases, expected_count in (
        (ROUTE_DEV_CASES, 8),
        (ROUTE_VALIDATION_CASES, 20),
        (ROUTE_BLIND_CASES, 20),
    ):
        counts = Counter(case.expected_route for case in cases)
        assert set(counts) == {route.value for route in RouteCategory}
        assert set(counts.values()) == {expected_count}


def test_evaluation_splits_do_not_reuse_templates_or_each_other():
    templates = {text for values in _TEMPLATES.values() for text in values}
    few_shots = {text for text, _ in _LLM_FEW_SHOTS}
    dev_messages = {case.message for case in ROUTE_DEV_CASES}
    validation_messages = {case.message for case in ROUTE_VALIDATION_CASES}
    blind_messages = {case.message for case in ROUTE_BLIND_CASES}

    for messages in (dev_messages, validation_messages, blind_messages):
        assert messages.isdisjoint(templates)
        assert messages.isdisjoint(few_shots)
    assert dev_messages.isdisjoint(validation_messages)
    assert dev_messages.isdisjoint(blind_messages)
    assert validation_messages.isdisjoint(blind_messages)
