
from core.supervisor import Supervisor

research_goal = (
    "Can specialized AI agents improve the reliability "
    "of scientific hypothesis generation?"
)

print("\nRESEARCH GOAL:")
print(research_goal)

supervisor = Supervisor(
    research_goal=research_goal,
    population_size=2,
)

state = supervisor.run(generations=2)

print("\n" + "=" * 60)
print("FINAL GENERATED HYPOTHESES")
print("=" * 60)

for hypothesis in state.hypotheses:
    print(f"\nID: {hypothesis.id}")
    print(f"Title: {hypothesis.title}")
    print(f"Elo Rating: {hypothesis.elo_rating}")
    print(f"Generation: {hypothesis.generation}")
    print(f"Description: {hypothesis.description}")
    print(f"Testable Prediction: {hypothesis.testable_prediction}")
