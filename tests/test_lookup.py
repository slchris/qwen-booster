from qwen_booster.lookup import accepted_prefix, draft_from_history


def test_lookup_uses_observed_continuation():
    assert draft_from_history([1, 2, 3, 4, 5, 9, 1, 2, 3], max_draft=2) == [4, 5]
    assert draft_from_history([1, 2, 3, 4, 5]) == []
    assert draft_from_history([1, 2, 3, 1, 2, 3], max_draft=0) == []


def test_acceptance_stops_at_first_rejection():
    assert accepted_prefix([1, 2, 3], [1, 9, 3, 4]) == 1
    assert accepted_prefix([1, 2], [1, 2, 3]) == 2
    assert accepted_prefix([1], [9, 3]) == 0
    assert accepted_prefix([], [9]) == 0
