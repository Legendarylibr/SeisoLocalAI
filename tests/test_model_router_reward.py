"""Tests for routing reward function."""

from seiso.model_router.reward import compute_route_reward


def test_reward_success_with_quality():
    reward = compute_route_reward(
        500.0,
        True,
        response_text="Here is a detailed explanation with enough content for the user.",
        classified_domain="code",
        route_domain_hints=("code",),
        complexity_score=0.5,
    )
    assert reward > 0.5


def test_reward_failure():
    assert compute_route_reward(100.0, False) == -1.0


def test_reward_empty_response():
    reward = compute_route_reward(100.0, True, response_text="", complexity_score=0.8)
    assert reward < 0.5
