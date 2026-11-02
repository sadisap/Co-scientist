
from pydantic import BaseModel, Field

from core.llm import generate_structured


class HypothesisScore(BaseModel):
    hypothesis_id: str

    grounding: int = Field(ge=1, le=5)
    testability: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    reasoning: int = Field(ge=1, le=5)

    explanation: str


class ScoreBatch(BaseModel):
    scores: list[HypothesisScore]


RUBRIC = """
You are evaluating scientific research hypotheses.

Score each hypothesis from 1 to 5 on:

GROUNDING
1 = unsupported or contradicts the provided literature
3 = partly supported, with some assumptions
5 = clearly supported by relevant retrieved evidence

TESTABILITY
1 = cannot be tested as written
3 = testable but missing important details
5 = clear variables, measurements, and comparison

FEASIBILITY
1 = unrealistic or impractical
3 = possible but significant details are missing
5 = realistic experiment with clear resources and methods

REASONING
1 = weak or unsupported reasoning
3 = reasonable but contains gaps
5 = clear reasoning that follows from the evidence

Do not reward long answers or scientific-sounding language.

Judge only the hypothesis and the provided literature.
Do not assume cited claims are true unless supported by that literature.

Return exactly one score for each hypothesis.
"""


def evaluate_blind(hypotheses, papers):
    paper_text = "\n\n".join(
        f"TITLE: {p.title}\nABSTRACT: {p.abstract}"
        for p in papers
    )

    hypothesis_text = "\n\n".join(
        f"""
ID: {h.id}
TITLE: {h.title}
DESCRIPTION: {h.description}
REASONING: {h.reasoning}
PREDICTION: {h.testable_prediction}
"""
        for h in hypotheses
    )

    return generate_structured(
        system_prompt=RUBRIC,
        user_prompt=(
            f"LITERATURE:\n{paper_text}\n\n"
            f"HYPOTHESES:\n{hypothesis_text}"
        ),
        response_model=ScoreBatch,
    )
