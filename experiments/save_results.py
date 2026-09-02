import json
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results")


def save_experiment(results: dict) -> Path:
    """Save an experiment run as JSON."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = RESULTS_DIR / (
        f"experiment_{timestamp}.json"
    )

    serializable = {
        "baseline_score": results["baseline_score"],
        "multi_agent_score": results["multi_agent_score"],

        "baseline_hypotheses": [
            hypothesis.model_dump()
            for hypothesis
            in results["baseline"].hypotheses
        ],

        "baseline_evaluations": [
            evaluation.model_dump()
            for evaluation
            in results[
                "baseline_evaluation"
            ].evaluations
        ],

        "multi_agent_hypotheses": [
            hypothesis.model_dump()
            for hypothesis
            in results[
                "multi_agent_state"
            ].hypotheses
        ],

        "multi_agent_evaluations": [
            evaluation.model_dump()
            for evaluation
            in results[
                "multi_agent_evaluation"
            ].evaluations
        ],

        "roadmap": (
            results["multi_agent_state"]
            .roadmap.model_dump()
            if results["multi_agent_state"].roadmap
            else None
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable,
            file,
            indent=2,
        )

    return output_path