"""Unit tests for investigation state."""

from backend.app.models.state import (
    ConfidenceBreakdown,
    InvestigationState,
    Observation,
    ObservationSource,
)


def test_confidence_recompute():
    c = ConfidenceBreakdown(visual=0.9, semantic=0.8, cross_view=0.7, retrieval_relevance=0.6)
    c.recompute()
    assert 0.0 <= c.overall <= 1.0
    assert c.level.value in ("high", "moderate", "low")


def test_state_summary_for_llm():
    s = InvestigationState()
    s.add_observation(
        Observation(source=ObservationSource.VISION, content="test fact", is_fact=True)
    )
    summary = s.summary_for_llm()
    assert "observations" in summary
    assert summary["iteration"] == 0


def test_can_continue_limits():
    s = InvestigationState(max_iterations=2)
    assert s.can_continue()
    s.iteration = 2
    assert not s.can_continue()
    s.iteration = 0
    s.status = "decided"
    assert not s.can_continue()
