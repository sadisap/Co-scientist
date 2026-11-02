
import json
import random
from pathlib import Path
from statistics import mean

from core.supervisor import Supervisor
from core.usage import reset_usage, get_usage
from experiments.baseline import run_baseline
from experiments.blind_evaluator import evaluate_blind
from models.hypothesis import Paper


RESULTS_DIR = Path("results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_question(question_id):
    data = json.loads(
        Path(f"benchmark/{question_id}.json").read_text()
    )

    papers = [Paper(**p) for p in data["papers"]]

    return data["research_goal"], papers


def run_trial(question_id, trial_number):
    question, papers = load_question(question_id)

    print(f"\n{question_id}, trial {trial_number}")

    reset_usage()

    baseline = run_baseline(
        research_goal=question,
        papers=papers,
        count=2,
    )

    baseline_usage = get_usage()

    reset_usage()

    supervisor = Supervisor(
        research_goal=question,
        population_size=2,
    )

    supervisor.state.papers = papers

    multi_state = supervisor.run(
        generations=2,
        retrieve_literature=False,
    )

    multi_usage = get_usage()

    # Randomize the order and remove architecture labels.
    candidates = []

    for h in baseline.hypotheses:
        candidates.append(("baseline", h))

    for h in multi_state.hypotheses:
        candidates.append(("multi_agent", h))

    rng = random.Random(f"{question_id}-{trial_number}")
    rng.shuffle(candidates)

    # Give every candidate a neutral ID.
    blinded = []

    for index, (_, hypothesis) in enumerate(candidates):
        h = hypothesis.model_copy(deep=True)
        h.id = f"Candidate_{index + 1}"
        blinded.append(h)

    reset_usage()
    evaluations = evaluate_blind(blinded, papers)
    evaluation_usage = get_usage()

    score_lookup = {
        score.hypothesis_id: score
        for score in evaluations.scores
    }

    records = []

    for (architecture, hypothesis), blinded_h in zip(
        candidates, blinded
    ):
        score = score_lookup[blinded_h.id]

        overall = mean([
            score.grounding,
            score.testability,
            score.feasibility,
            score.reasoning,
        ])

        records.append({
            "architecture": architecture,
            "hypothesis": hypothesis.model_dump(),
            "scores": score.model_dump(),
            "overall": overall,
        })

    result = {
        "question_id": question_id,
        "trial": trial_number,
        "model_provider": "openai",
        "records": records,
        "baseline_usage": baseline_usage,
        "multi_agent_usage": multi_usage,
        "evaluation_usage": evaluation_usage,
    }

    path = RESULTS_DIR / (
        f"{question_id}_trial_{trial_number}.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    baseline_scores = [
        r["overall"] for r in records
        if r["architecture"] == "baseline"
    ]

    multi_scores = [
        r["overall"] for r in records
        if r["architecture"] == "multi_agent"
    ]

    print("Baseline:", round(mean(baseline_scores), 2))
    print("Multi-agent:", round(mean(multi_scores), 2))
    print("Saved:", path)


if __name__ == "__main__":
    for question_id in ("Q1", "Q2", "Q3"):
        for trial in range(1, 4):
            run_trial(question_id, trial)
