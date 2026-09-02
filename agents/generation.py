from core.llm import generate_structured
from models.hypothesis import HypothesisBatch, Paper


GENERATION_SYSTEM_PROMPT = """
You are the Generation Agent in a multi-agent AI scientific research system.

Your job is to generate diverse, scientifically meaningful hypotheses
for a given research goal.

Each hypothesis should:

1. Address the research goal directly.
2. Make a specific scientific claim.
3. Include reasoning explaining why the hypothesis may be true.
4. Include any supporting evidence or prior knowledge you know.
5. Include possible contradictory evidence or weaknesses.
6. Produce a concrete, testable prediction.
7. Be meaningfully different from the other hypotheses.

Do not simply paraphrase the same idea multiple ways.

Prefer hypotheses that are:
- specific
- falsifiable
- scientifically useful
- experimentally testable

You are generating candidate ideas, not deciding which one is best.
Other agents will critique and rank them later.
"""


def generate_hypotheses(
    research_goal: str,
    count: int = 3,
    papers: list[Paper] | None = None,
) -> HypothesisBatch:
    """Generate an initial population of scientific hypotheses."""

    literature_context = ""

    if papers:
        literature_sections = []

        for paper in papers:
            literature_sections.append(
                f"""
TITLE: {paper.title}
YEAR: {paper.publication_year}
DOI: {paper.doi}
CITATIONS: {paper.cited_by_count}

ABSTRACT:
{paper.abstract}
"""
            )

        literature_context = (
            "\n\nRETRIEVED LITERATURE:\n"
            + "\n".join(literature_sections)
        )

    user_prompt = f"""
Research goal:

{research_goal}

{literature_context}

Generate exactly {count} distinct candidate hypotheses.

Assign IDs:
H1, H2, H3, and so on.

All hypotheses belong to generation 0.
Their initial Elo rating should be 1000.

IMPORTANT:
When retrieved literature is provided, ground your reasoning in that
literature.

Do not invent papers, citations, authors, statistics, or experimental
results.

If the retrieved literature does not support a factual claim, describe
the claim as speculation rather than established evidence.
"""

    return generate_structured(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=HypothesisBatch,
    )