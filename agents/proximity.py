from core.llm import generate_structured
from models.hypothesis import Hypothesis, ProximityResult


PROXIMITY_SYSTEM_PROMPT = """
You are the Proximity Agent in a multi-agent scientific research system.

Your job is to detect conceptual similarity between scientific hypotheses.

For every pair of hypotheses, estimate how similar their underlying
scientific ideas are.

Score similarity from:

0.0 = completely different scientific ideas
1.0 = essentially the same hypothesis expressed differently

Focus on conceptual similarity, NOT wording similarity.

Two hypotheses may use different terminology while still representing
the same underlying scientific idea.
"""


def analyze_proximity(
    hypotheses: list[Hypothesis],
) -> ProximityResult:

    hypothesis_text = "\n\n".join(
        [
            f"""
ID: {h.id}
TITLE: {h.title}
DESCRIPTION: {h.description}
"""
            for h in hypotheses
        ]
    )

    user_prompt = f"""
Analyze conceptual similarity between every unique pair of hypotheses.

HYPOTHESES:

{hypothesis_text}

Return exactly one similarity assessment for every unique pair.
"""

    return generate_structured(
        system_prompt=PROXIMITY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ProximityResult,
    )