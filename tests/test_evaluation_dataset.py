from collections import Counter

from core.intent_recognizer import RouteCategory, _TEMPLATES
from evaluation.evaluator import DEFAULT_ROUTE_CASES


def test_route_dev_dataset_covers_each_route_evenly():
    counts = Counter(case.expected_route for case in DEFAULT_ROUTE_CASES)

    assert len(DEFAULT_ROUTE_CASES) == 24
    assert set(counts) == {route.value for route in RouteCategory}
    assert set(counts.values()) == {8}


def test_route_dev_messages_are_not_few_shot_templates():
    templates = {text for values in _TEMPLATES.values() for text in values}
    messages = {case.message for case in DEFAULT_ROUTE_CASES}

    assert len(messages) == len(DEFAULT_ROUTE_CASES)
    assert messages.isdisjoint(templates)
