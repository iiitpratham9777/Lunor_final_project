#!/usr/bin/env python3
"""
End-to-end demo of the Multimodal AI Investigation Agent.
Runs without external API keys using OpenCV/mock vision + local retrieval.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agents.orchestrator import Orchestrator
from backend.app.core.logging import setup_logging
from backend.app.models.state import InvestigationState
from backend.app.services.seed import seed_demo_knowledge
from backend.app.evaluation.runner import default_adversarial_cases, run_evaluation


async def main():
    setup_logging()
    seed_demo_knowledge()

    demo_dir = ROOT / "tests" / "demo_data"
    images = sorted(demo_dir.glob("anomaly_*.png"))
    if not images:
        images = sorted(demo_dir.glob("*.png"))

    print("=" * 60)
    print("Multimodal AI Investigation Agent — DEMO")
    print("=" * 60)

    state = InvestigationState(
        input_images=[str(p) for p in images[:2]],
        textual_description="Inspect object for surface defects or scratches",
        metadata={"demo": True},
    )

    orch = Orchestrator()
    state = await orch.run(state)

    print(f"\nInvestigation ID : {state.investigation_id}")
    print(f"Status           : {state.status}")
    print(f"Iterations       : {state.iteration}")
    print(f"Decision         : {state.final_decision}")
    print(f"Confidence       : {state.confidence.overall:.3f} ({state.confidence.level.value})")
    print(f"Latency          : {state.latency.get('total_ms', 0):.0f} ms")
    print(f"Tool calls       : {len(state.tool_calls)}")
    print(f"\n--- Trace ---")
    for t in state.trace:
        print(f"  [{t['iteration']}] {t['node']} → {t['action']}")
    print(f"\n--- Explanation ---\n{state.explanation}")

    print("\n" + "=" * 60)
    print("Running evaluation suite (adversarial + demo cases)...")
    print("=" * 60)
    cases = default_adversarial_cases(demo_dir)
    report = await run_evaluation(cases)
    print(f"Cases            : {report.n_cases}")
    print(f"Accuracy         : {report.accuracy}")
    print(f"Uncertainty rate : {report.uncertainty_rate:.2f}")
    print(f"Avg iterations   : {report.avg_iterations:.1f}")
    print(f"Avg latency ms   : {report.avg_latency_ms:.0f}")
    print(f"Avg tool calls   : {report.avg_tool_calls:.1f}")
    for r in report.results:
        mark = "✓" if r.correct else ("?" if r.correct is None else "✗")
        print(f"  {mark} {r.name}: pred={r.predicted} conf={r.confidence:.2f} iters={r.iterations}")

    print("\nDemo complete.")


if __name__ == "__main__":
    asyncio.run(main())

