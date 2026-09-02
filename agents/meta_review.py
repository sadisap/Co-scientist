from core.llm import generate_structured
from models.hypothesis import (
    Hypothesis,
    HypothesisReview,
    ResearchRoadmap,
)


META_REVIEW_SYSTEM_PROMPT = """
You are the Meta-Review Agent in a multi-agent AI scientific research system.

Your job is to synthesize the entire scientific reasoning process into
a final research roadmap.

You are given:
- the research goal
- the final ranked hypotheses
- peer reviews produced during the process

Your job is NOT simply to select the hypothesis with the highest Elo score.

Instead:

1. Identify the strongest overall hypothesis.
2. Explain why it is scientifically promising.
3. Preserve important criticisms raised by reviewers.
4. Identify meaningful alternative hypotheses.
5. Identify unresolved scientific questions.
6. Recommend a concrete experiment that could test the leading hypothesis.
7. Give a balanced overall assessment.

Do not invent citations, evidence, experimental results, or statistics.

Clearly distinguish between:
- evidence
- reasoning
- speculation

A high Elo rating means the hypothesis performed well in pairwise
comparisons. It does NOT prove the hypothesis is scientifically correct.
"""


def create_research_roadmap(
    research_goal: str,
    hypotheses: list[Hypothesis],
    reviews: list[HypothesisReview],
) -> ResearchRoadmap:
    """Synthesize the research process into a final roadmap."""

    hypothesis_text = "\n\n".join(
        [
            f"""
ID: {h.id}
TITLE: {h.title}
DESCRIPTION: {h.description}
REASONING: {h.reasoning}
TESTABLE PREDICTION: {h.testable_prediction}
ELO: {h.elo_rating}
GENERATION: {h.generation}
"""
            for h in hypotheses
        ]
    )

    review_text = "\n\n".join(
        [
            f"""
HYPOTHESIS: {review.hypothesis_id}
PLAUSIBILITY: {review.plausibility_score}
NOVELTY: {review.novelty_score}
TESTABILITY: {review.testability_score}
STRENGTHS: {review.strengths}
WEAKNESSES: {review.weaknesses}
MAJOR CONCERNS: {review.major_concerns}
RECOMMENDATION: {review.recommendation}
"""
            for review in reviews
        ]
    )

    user_prompt = f"""
RESEARCH GOAL:

{research_goal}


FINAL HYPOTHESES:

{hypothesis_text}


REVIEWS FROM THE RESEARCH PROCESS:

{review_text}


Produce the final research roadmap.
"""

    return generate_structured(
        system_prompt=META_REVIEW_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ResearchRoadmap,
    )