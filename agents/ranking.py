from core.llm import generate_structured
from models.hypothesis import Hypothesis, PairwiseDecision


RANKING_SYSTEM_PROMPT = """
You are the Ranking Agent in a multi-agent AI scientific research system.

You will receive two scientific hypotheses addressing the same research goal.

Your job is to determine which is the stronger research proposal.

Compare them based on:

1. Scientific plausibility
2. Novelty
3. Testability
4. Quality of reasoning
5. Strength of evidence
6. Scientific usefulness

You MUST select exactly one winner.

Do not declare a tie.

Do not rewrite the hypotheses.
"""


def compare_hypotheses(
    hypothesis_a: Hypothesis,
    hypothesis_b: Hypothesis,
) -> PairwiseDecision:

    user_prompt = f"""
HYPOTHESIS A

ID: {hypothesis_a.id}
Title: {hypothesis_a.title}
Description: {hypothesis_a.description}
Reasoning: {hypothesis_a.reasoning}
Prediction: {hypothesis_a.testable_prediction}


HYPOTHESIS B

ID: {hypothesis_b.id}
Title: {hypothesis_b.title}
Description: {hypothesis_b.description}
Reasoning: {hypothesis_b.reasoning}
Prediction: {hypothesis_b.testable_prediction}


Select the stronger hypothesis.

winner_id MUST be either:
{hypothesis_a.id}
or
{hypothesis_b.id}

loser_id MUST be the other hypothesis.
"""

    return generate_structured(
        system_prompt=RANKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=PairwiseDecision,
    )


def update_elo(
    winner: Hypothesis,
    loser: Hypothesis,
    k_factor: float = 32.0,
) -> None:
    """Update Elo ratings after a pairwise comparison."""

    expected_winner = 1 / (
        1 + 10 ** ((loser.elo_rating - winner.elo_rating) / 400)
    )

    expected_loser = 1 / (
        1 + 10 ** ((winner.elo_rating - loser.elo_rating) / 400)
    )

    winner.elo_rating += k_factor * (1 - expected_winner)
    loser.elo_rating += k_factor * (0 - expected_loser)


def run_tournament(
    hypotheses: list[Hypothesis],
) -> list[Hypothesis]:
    """
    Run a round-robin tournament.

    Every hypothesis competes against every other hypothesis once.
    """

    for i in range(len(hypotheses)):
        for j in range(i + 1, len(hypotheses)):

            hypothesis_a = hypotheses[i]
            hypothesis_b = hypotheses[j]

            decision = compare_hypotheses(
                hypothesis_a,
                hypothesis_b,
            )

            if decision.winner_id == hypothesis_a.id:
                winner = hypothesis_a
                loser = hypothesis_b

            elif decision.winner_id == hypothesis_b.id:
                winner = hypothesis_b
                loser = hypothesis_a

            else:
                raise ValueError(
                    f"Ranking agent returned invalid winner ID: "
                    f"{decision.winner_id}"
                )

            update_elo(winner, loser)

    return sorted(
        hypotheses,
        key=lambda hypothesis: hypothesis.elo_rating,
        reverse=True,
    )