from core.llm import generate_structured
from models.hypothesis import Hypothesis, EvolutionBatch


EVOLUTION_SYSTEM_PROMPT = """
You are the Evolution Agent in a multi-agent AI scientific research system.

Your job is to create improved scientific hypotheses from strong existing
hypotheses.

You may improve hypotheses using two strategies:

1. REFINEMENT
   Improve a hypothesis by:
   - making it more specific
   - fixing weaknesses
   - improving testability
   - removing unsupported claims
   - strengthening the reasoning

2. COMBINATION
   Combine useful aspects of different hypotheses when doing so creates
   a genuinely stronger and coherent new research idea.

Do not merely paraphrase the originals.

The evolved hypotheses should represent meaningful improvements.

Each evolved hypothesis must contain:
- a clear scientific claim
- reasoning
- supporting evidence if available
- contradictory evidence or limitations
- a testable prediction

Do not invent citations.
If reliable evidence is unavailable, leave supporting evidence empty.
"""


def evolve_hypotheses(
    hypotheses: list[Hypothesis],
    count: int = 2,
    generation: int = 1,
) -> EvolutionBatch:

    hypothesis_text = "\n\n".join(
        [
            f"""
ID: {h.id}
TITLE: {h.title}

DESCRIPTION:
{h.description}

REASONING:
{h.reasoning}

TESTABLE PREDICTION:
{h.testable_prediction}

ELO:
{h.elo_rating}
"""
            for h in hypotheses
        ]
    )

    user_prompt = f"""
Here are the strongest hypotheses from the current population:

{hypothesis_text}

Create exactly {count} evolved hypotheses.

Use IDs beginning with E{generation}_.

For example:
E{generation}_1
E{generation}_2

Set generation={generation}.

Set initial Elo rating=1000.

Produce genuinely improved hypotheses rather than simple rewrites.
"""

    return generate_structured(
        system_prompt=EVOLUTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=EvolutionBatch,
    )