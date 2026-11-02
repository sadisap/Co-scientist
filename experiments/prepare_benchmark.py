
import json
from pathlib import Path

from tools.literature import search_literature

QUESTIONS = {
    "Q1": "How can AI systems reduce hallucinations when generating scientific hypotheses?",
    "Q2": "Can evidence verification improve the reliability of LLM-generated scientific hypotheses?",
    "Q3": "Does iterative critique improve the testability of LLM-generated scientific hypotheses?",
}

output_dir = Path("benchmark")
output_dir.mkdir(exist_ok=True)

for question_id, question in QUESTIONS.items():
    print(f"\nRetrieving literature for {question_id}")

    literature = search_literature(
        query=question.rstrip("?.! "),
        limit=5,
    )

    output = {
        "question_id": question_id,
        "research_goal": question,
        "papers": [
            paper.model_dump()
            for paper in literature.papers
        ],
    }

    path = output_dir / f"{question_id}.json"

    path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(literature.papers)} papers to {path}")
