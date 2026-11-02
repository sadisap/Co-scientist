
import json
from pathlib import Path

from experiments.multi_agent import run_comparison

research_goal = (
    "How can AI systems reduce hallucinations "
    "when generating scientific hypotheses?"
)

print("\nOPENAI ARCHITECTURE COMPARISON")
print("Research goal:", research_goal)

results = run_comparison(
    research_goal=research_goal,
    hypothesis_count=2,
    generations=2,
)

print("\nRESULTS")
print("Single-agent score:", results["baseline_score"])
print("Multi-agent score:", results["multi_agent_score"])

difference = (
    results["multi_agent_score"]
    - results["baseline_score"]
)

print("Difference:", difference)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output = {
    "research_goal": research_goal,
    "baseline_score": results["baseline_score"],
    "multi_agent_score": results["multi_agent_score"],
    "difference": difference,
    "baseline_hypotheses": [
        h.model_dump()
        for h in results["baseline"].hypotheses
    ],
    "multi_agent_hypotheses": [
        h.model_dump()
        for h in results["multi_agent_state"].hypotheses
    ],
    "baseline_evaluations": [
        e.model_dump()
        for e in results["baseline_evaluation"].evaluations
    ],
    "multi_agent_evaluations": [
        e.model_dump()
        for e in results["multi_agent_evaluation"].evaluations
    ],
}

output_path = output_dir / "openai_comparison.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=2)

print("\nSaved results to:", output_path)
