from tools.literature import search_literature
from experiments.baseline import run_baseline


research_goal = (
    "Investigate whether multiple specialized LLM agents "
    "can produce better research hypotheses than a single LLM."
)

print("\nResearch Goal:")
print(research_goal)

print("\nSearching OpenAlex...")

literature = search_literature(
    query=research_goal,
    limit=5,
)

print(f"\nRetrieved {len(literature.papers)} papers:")

for i, paper in enumerate(literature.papers, start=1):
    print(f"\nPaper {i}")
    print(f"Title: {paper.title}")
    print(f"Year: {paper.publication_year}")
    print(f"DOI: {paper.doi}")
    print(f"Abstract: {paper.abstract[:300]}...")

print("\nGenerating hypotheses using retrieved literature...")

result = run_baseline(
    research_goal=research_goal,
    papers=literature.papers,
    count=2,
)

print("\nGenerated Hypotheses:")

for hypothesis in result.hypotheses:
    print(f"\n{hypothesis.id}: {hypothesis.title}")
    print(f"Description: {hypothesis.description}")
    print(f"Reasoning: {hypothesis.reasoning}")
    print(f"Supporting Evidence: {hypothesis.supporting_evidence}")
    print(f"Testable Prediction: {hypothesis.testable_prediction}")