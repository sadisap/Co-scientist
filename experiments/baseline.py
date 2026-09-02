from core.llm import generate_structured
from models.hypothesis import HypothesisBatch, Paper


BASELINE_SYSTEM_PROMPT = """
You are an AI scientific research assistant.

Given a research goal and relevant scientific literature, generate
high-quality scientific hypotheses.

Each hypothesis should:
- make a specific scientific claim
- contain clear reasoning
- be scientifically plausible
- be meaningfully novel
- be experimentally testable
- include a concrete testable prediction

Do not invent citations, papers, statistics, or experimental results.

If evidence is unavailable, clearly describe claims as speculation.
"""


def run_baseline(
    research_goal: str,
    papers: list[Paper],
    count: int = 3,
) -> HypothesisBatch:
    """Generate hypotheses using one direct LLM call."""

    literature_text = "\n\n".join(
        [
            f"""
TITLE: {paper.title}
YEAR: {paper.publication_year}
DOI: {paper.doi}

ABSTRACT:
{paper.abstract}
"""
            for paper in papers
        ]
    )

    user_prompt = f"""
RESEARCH GOAL:

{research_goal}


RETRIEVED LITERATURE:

{literature_text}


Generate exactly {count} distinct hypotheses.

Use IDs:
B1, B2, B3, and so on.

Set generation=0.
Set Elo rating=1000.
"""

    return generate_structured(
        system_prompt=BASELINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=HypothesisBatch,
    )