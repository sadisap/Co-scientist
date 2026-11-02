from core.llm import generate_structured
from models.hypothesis import HypothesisBatch
from experiments.baseline import BASELINE_SYSTEM_PROMPT


research_goal = (
    "Investigate whether multiple specialized LLM agents "
    "can produce better research hypotheses than a single LLM."
)

user_prompt = f"""
RESEARCH GOAL:

{research_goal}

Generate exactly 2 distinct scientific hypotheses.

Use IDs B1 and B2.
Set generation=0.
Set Elo rating=1000.
"""

result = generate_structured(
    system_prompt=BASELINE_SYSTEM_PROMPT,
    user_prompt=user_prompt,
    response_model=HypothesisBatch,
)

print("\nResearch Goal:")
print(research_goal)

print("\nGenerated Hypotheses:")

for hypothesis in result.hypotheses:
    print(f"\n{hypothesis.id}: {hypothesis.title}")
    print(f"Description: {hypothesis.description}")
    print(f"Reasoning: {hypothesis.reasoning}")
    print(f"Testable Prediction: {hypothesis.testable_prediction}")