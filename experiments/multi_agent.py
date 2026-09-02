from statistics import mean

from core.supervisor import Supervisor
from experiments.baseline import run_baseline
from experiments.evaluator import evaluate_hypotheses
from tools.literature import search_literature


def average_score(evaluations) -> float:
    if not evaluations:
        return 0.0

    return mean(
        evaluation.overall_score
        for evaluation in evaluations
    )


def run_comparison(
    research_goal: str,
    hypothesis_count: int = 3,
    generations: int = 2,
):
    """
    Compare a single-agent baseline against the multi-agent system.
    """

    print("\n=== EXPERIMENT ===")

    print("\nRetrieving shared literature...")

    literature = search_literature(
        query=research_goal,
        limit=5,
    )

    print(
        f"Retrieved {len(literature.papers)} papers."
    )

    # -------------------------
    # Baseline
    # -------------------------

    print("\nRunning single-agent baseline...")

    baseline = run_baseline(
        research_goal=research_goal,
        papers=literature.papers,
        count=hypothesis_count,
    )

    print("\nEvaluating baseline...")

    baseline_evaluation = evaluate_hypotheses(
        baseline.hypotheses
    )

    # -------------------------
    # Multi-agent
    # -------------------------

    print("\nRunning multi-agent system...")

    supervisor = Supervisor(
        research_goal=research_goal,
        population_size=hypothesis_count,
    )

    # Reuse the same literature so the comparison is fair.
    supervisor.state.papers = literature.papers

    final_state = supervisor.run(
        generations=generations,
        retrieve_literature=False,
    )

    print("\nEvaluating multi-agent hypotheses...")

    multi_evaluation = evaluate_hypotheses(
        final_state.hypotheses
    )

    baseline_score = average_score(
        baseline_evaluation.evaluations
    )

    multi_score = average_score(
        multi_evaluation.evaluations
    )

    return {
        "baseline": baseline,
        "baseline_evaluation": baseline_evaluation,
        "baseline_score": baseline_score,

        "multi_agent_state": final_state,
        "multi_agent_evaluation": multi_evaluation,
        "multi_agent_score": multi_score,
    }