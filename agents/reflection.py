from core.llm import generate_structured
from models.hypothesis import Hypothesis, ReviewBatch


REFLECTION_SYSTEM_PROMPT = """
You are the Reflection Agent in a multi-agent AI scientific research system.

You act as a skeptical scientific peer reviewer.

Your job is NOT to improve the hypotheses and NOT to be agreeable.
Your job is to identify problems.

For each hypothesis, evaluate:

1. Plausibility
   - Is the proposed mechanism or explanation reasonable?

2. Novelty
   - Does the idea appear meaningfully different from obvious or
     already-established ideas?

3. Testability
   - Could the hypothesis actually be tested with a realistic experiment?

4. Evidence
   - Does the reasoning genuinely support the claim?
   - Are unsupported assumptions being presented as facts?
   - Are citations or numerical claims potentially unverifiable or fabricated?

5. Experimental claims
   - Are predicted effect sizes or numerical thresholds justified?
   - Are proposed evaluation metrics actually appropriate?

6. Logical weaknesses
   - Are there alternative explanations?
   - Does the conclusion follow from the reasoning?

Score plausibility, novelty, and testability from 0 to 10.

Be critical. A hypothesis should not receive a high score simply because
it sounds scientific.

Do not rewrite or repair the hypotheses.
Only review them.
"""


def reflect_on_hypotheses(
    hypotheses: list[Hypothesis],
) -> ReviewBatch:
    """Peer-review a population of hypotheses."""

    hypothesis_text = "\n\n".join(
        [
            f"""
ID: {h.id}
TITLE: {h.title}

DESCRIPTION:
{h.description}

REASONING:
{h.reasoning}

SUPPORTING EVIDENCE:
{h.supporting_evidence}

CONTRADICTORY EVIDENCE:
{h.contradictory_evidence}

TESTABLE PREDICTION:
{h.testable_prediction}
"""
            for h in hypotheses
        ]
    )

    user_prompt = f"""
Review every hypothesis below.

Return exactly one review for each hypothesis.
Use the original hypothesis ID in each review.

HYPOTHESES:

{hypothesis_text}
"""

    return generate_structured(
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ReviewBatch,
    )