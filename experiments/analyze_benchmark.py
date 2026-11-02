
import json
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path("results/benchmark")

DIMENSIONS = [
    "grounding",
    "testability",
    "feasibility",
    "reasoning",
]


def load_results():
    results = []

    for path in sorted(RESULTS_DIR.glob("Q*_trial_*.json")):
        with path.open(encoding="utf-8") as file:
            results.append(json.load(file))

    return results


def average(values):
    return mean(values) if values else 0.0


def main():
    results = load_results()

    print("\nBENCHMARK ANALYSIS")
    print("=" * 60)
    print("Completed trials:", len(results))

    if len(results) != 9:
        print("WARNING: Expected 9 completed trials.")

    for architecture in ["baseline", "multi_agent"]:
        records = [
            record
            for result in results
            for record in result["records"]
            if record["architecture"] == architecture
        ]

        print(f"\n{architecture.upper()}")
        print("-" * 40)

        overall = [r["overall"] for r in records]

        print("Hypotheses evaluated:", len(records))
        print("Average overall:", round(average(overall), 3))

        if len(overall) > 1:
            print("Standard deviation:", round(stdev(overall), 3))

        for dimension in DIMENSIONS:
            scores = [
                r["scores"][dimension]
                for r in records
            ]

            print(
                f"{dimension}:",
                round(average(scores), 3)
            )

        usage_key = (
            "baseline_usage"
            if architecture == "baseline"
            else "multi_agent_usage"
        )

        print("\nResource usage:")

        for key in [
            "calls",
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
        ]:
            values = [
                result[usage_key][key]
                for result in results
            ]

            print(
                f"Average {key}:",
                round(average(values), 2)
            )

    print("\nPER-QUESTION RESULTS")
    print("-" * 60)

    for question_id in ["Q1", "Q2", "Q3"]:
        question_results = [
            result
            for result in results
            if result["question_id"] == question_id
        ]

        for architecture in ["baseline", "multi_agent"]:
            scores = [
                record["overall"]
                for result in question_results
                for record in result["records"]
                if record["architecture"] == architecture
            ]

            print(
                question_id,
                architecture,
                round(average(scores), 3),
            )


if __name__ == "__main__":
    main()
