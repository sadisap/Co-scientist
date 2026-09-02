from experiments.multi_agent import run_comparison
from experiments.save_results import save_experiment


def main():
    research_goal = (
        "Investigate whether multiple specialized LLM agents "
        "can produce better research hypotheses than a single LLM."
    )

    print("\n" + "=" * 70)
    print("MINI AI CO-SCIENTIST")
    print("=" * 70)

    print(f"\nResearch goal:\n{research_goal}")

    results = run_comparison(
        research_goal=research_goal,
        hypothesis_count=2,
        generations=2,
    )

    print("\n\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        f"\nSingle-agent average score: "
        f"{results['baseline_score']:.2f}/10"
    )

    print(
        f"Multi-agent average score:  "
        f"{results['multi_agent_score']:.2f}/10"
    )

    difference = (
        results["multi_agent_score"]
        - results["baseline_score"]
    )

    print(
        f"Difference:                 "
        f"{difference:+.2f}"
    )

    print("\n--- FINAL MULTI-AGENT HYPOTHESES ---")

    for rank, hypothesis in enumerate(
        results["multi_agent_state"].hypotheses,
        start=1,
    ):
        print(
            f"\n#{rank} "
            f"{hypothesis.title}"
        )
        print(
            f"Elo: {hypothesis.elo_rating:.2f}"
        )
        print(hypothesis.description)

    roadmap = results[
        "multi_agent_state"
    ].roadmap

    if roadmap:
        print("\n\n--- META-REVIEW ---")

        print(
            f"\nLeading hypothesis: "
            f"{roadmap.leading_hypothesis_id}"
        )

        print(
            f"\n{roadmap.leading_hypothesis_summary}"
        )

        print("\nRecommended experiment:")
        print(roadmap.recommended_experiment)

        print("\nOverall assessment:")
        print(roadmap.overall_assessment)

    output_path = save_experiment(results)

    print(
        f"\n\nResults saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()