# Mini AI Co-Scientist

A small-scale implementation of a multi-agent scientific reasoning system inspired by Google's AI Co-Scientist architecture.

The system takes a research goal, retrieves relevant literature, generates candidate hypotheses, critiques and ranks them, evolves promising candidates, and produces a final research roadmap.

## Architecture

```text
                         Research Goal
                              |
                         OpenAlex Search
                              |
                      Retrieved Literature
                              |
                         Supervisor
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
    Generation           Reflection            Proximity
        |                (peer review)         (similarity)
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                        Elo Tournament
                              |
                       Top Hypotheses
                              |
                              v
                          Evolution
                     (refine + combine)
                              |
                              +-----> repeat
                              |
                              v
                         Meta-Review
                              |
                              v
                      Research Roadmap
```

The current prototype uses Qwen locally through Ollama for agent reasoning and OpenAlex for literature retrieval.

### Agents

| Agent       | Role                                                                               |
| ----------- | ---------------------------------------------------------------------------------- |
| Generation  | Generates candidate hypotheses from the research goal and retrieved literature     |
| Reflection  | Reviews hypotheses for plausibility, novelty, testability, evidence, and reasoning |
| Ranking     | Compares hypotheses pairwise and updates their Elo ratings                         |
| Evolution   | Refines and combines highly ranked hypotheses                                      |
| Proximity   | Measures conceptual similarity between hypotheses                                  |
| Meta-Review | Synthesizes the final hypotheses and reviews into a research roadmap               |
| Supervisor  | Maintains shared state and coordinates the reasoning loop                          |

## Experiment

I wanted to test whether the multi-agent pipeline actually improved hypothesis quality rather than assuming that additional agents would help.

The comparison used the same research goal and retrieved literature for both conditions.

```text
                 Shared Research Goal
                         +
                 Shared Literature
                         |
              +----------+----------+
              |                     |
              v                     v
        Single Agent           Multi-Agent
              |                     |
        Generate Once        Generate
              |               Reflect
              |               Rank
              |               Evolve
              |               Repeat
              |                     |
              +----------+----------+
                         |
                         v
                      Evaluator
```

The first complete run produced:

| System                | Average Evaluator Score |
| --------------------- | ----------------------: |
| Single-agent baseline |               4.00 / 10 |
| Multi-agent system    |               3.00 / 10 |
| Difference            |                   -1.00 |

The multi-agent system performed worse in this run.

This is only one experiment with one research goal and a small local model, so the result is not enough to make a general claim about multi-agent systems. It did expose several problems worth investigating.

### What went wrong?

Hypothesis convergence
The evolved hypotheses became conceptually similar. Selecting and repeatedly refining the strongest candidates may have reduced exploration too quickly.

Passive proximity analysis
The system detects similarity between hypotheses, but the Supervisor does not currently use that information to change selection, remove duplicates, or encourage exploration.

Shared-model bias
The same underlying model generates, critiques, ranks, evolves, and evaluates hypotheses. The different agents have different roles, but they can still share the same underlying weaknesses.

Error amplification
Additional reasoning stages create opportunities to correct mistakes, but they also create opportunities to propagate them.

Model capability
The experiment uses a small local Qwen model. Scientific synthesis, novelty assessment, peer review, and experimental design may require a substantially stronger underlying model.

The experiment also leaves an important comparison unresolved: whether improvements from multi-agent systems come from the architecture itself or simply from using more test-time compute than a direct single-agent call.

The complete first experiment is saved in [`results/`](results/).

## Running Locally

Requirements:

* Python 3.11+
* Ollama

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the model configured in `core/llm.py` and start Ollama:

```bash
ollama pull qwen3:1.7b
ollama serve
```

Run the experiment:

```bash
python3 main.py
```

Experiment outputs are saved as JSON files under `results/`.

## Next Steps

* Use proximity scores during selection to preserve hypothesis diversity
* Compare single-agent and multi-agent systems under equal inference budgets
* Test stronger local and hosted models
* Separate generation and evaluation across different models
* Run the comparison across multiple research goals and repeated trials
* Evaluate hypotheses using stronger external or human evaluation

## Reference

The architecture is inspired by Google's publicly described AI Co-Scientist system. This project is an independent implementation built to explore the architecture and its failure modes.
