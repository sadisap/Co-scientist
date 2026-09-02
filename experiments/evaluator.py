from pydantic import BaseModel, Field

from core.llm import generate_structured
from models.hypothesis import Hypothesis


class HypothesisEvaluation(BaseModel):
    hypothesis_id: str

    plausibility: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    testability: float = Field(ge=0, le=10)
    specificity: float = Field(ge=0, le=10)
    reasoning_quality: float = Field(ge=0, le=10)

    overall_score: float = Field(ge=0, le=10)

    explanation: str


class EvaluationBatch(BaseModel):
    evaluations: list[HypothesisEvaluation]


EVALUATOR_SYSTEM_PROMPT = """
You are an evaluator of scientific hypotheses.

Evaluate each hypothesis independently.

Score each dimension from 0 to 10:

- plausibility
- novelty
- testability
- specificity
- reasoning quality
- overall quality

Be skeptical.

Do not reward:
- scientific-sounding language
- unsupported numerical claims
- fabricated evidence
- unnecessary complexity

Judge the quality of the scientific idea, not the writing style.
"""


def evaluate_hypotheses(
    hypotheses: list[Hypothesis],
) -> EvaluationBatch:

    hypothesis_text = "\n\n".join(
        [
            f"""
ID: {h.id}
TITLE: {h.title}
DESCRIPTION: {h.description}
REASONING: {h.reasoning}
PREDICTION: {h.testable_prediction}
"""
            for h in hypotheses
        ]
    )

    user_prompt = f"""
Evaluate every hypothesis below.

{hypothesis_text}

Return exactly one evaluation per hypothesis.
"""

    return generate_structured(
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=EvaluationBatch,
    )