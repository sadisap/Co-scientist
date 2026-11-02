# Mini AI Co-scientist

A small multi-agent system for generating and evaluating scientific research hypotheses, inspired by Google's AI Co-Scientist.

The project explores whether using specialized LLM agents produces better research hypotheses than a single-agent approach.

## How It Works

The system takes a research question, retrieves related scientific papers from OpenAlex, and generates hypotheses using an LLM.

The multi-agent workflow includes:

1. **Generation:** Creates hypotheses using the research question and retrieved papers.
2. **Reflection:** Reviews hypotheses for weaknesses, evidence, and testability.
3. **Proximity:** Measures similarity between hypotheses.
4. **Ranking:** Compares hypotheses using an Elo-based tournament.
5. **Evolution:** Refines or combines promising hypotheses.
6. **Meta-review:** Summarizes the final hypotheses into a research roadmap.

A Supervisor coordinates the workflow and maintains shared state.

The system supports local Qwen models through Ollama and OpenAI models through the API. Structured outputs are validated using Pydantic.

## Experiments

I compared the single-agent baseline with the multi-agent system using three research questions about LLM reliability and scientific hypothesis generation.

Both systems used the same underlying model and retrieved literature. Each question was tested three times, with two hypotheses generated per system in each trial.

Hypotheses were evaluated on scientific grounding, testability, feasibility, and reasoning using a fixed 1–5 rubric. Architecture labels were hidden during evaluation.

### Results

| Metric | Single Agent | Multi-Agent |
|---|---:|---:|
| Average quality score | 4.056 / 5 | 3.875 / 5 |
| Average model calls | 1 | 9 |
| Average input tokens | 1,723 | 9,766 |
| Average output tokens | 943 | 5,886 |
| Average runtime | 11.08s | 75.36s |

The single-agent baseline scored slightly higher overall while using considerably less computation. The multi-agent system scored slightly higher on reasoning, but lower on grounding, testability, and feasibility.

These results are limited to a small benchmark evaluated by an LLM. The comparison also does not use equal token budgets, so the results should not be treated as a general conclusion about multi-agent systems.

## Running Locally

Requires Python, Ollama for local inference, or an OpenAI API key for hosted inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the model provider in a `.env` file:

```text
MODEL_PROVIDER=openai
OPENAI_MODEL=gpt-6-luna
OPENAI_API_KEY=your_api_key
```

For local inference, set `MODEL_PROVIDER=ollama` and configure `OLLAMA_MODEL`.

Prepare the benchmark literature:

```bash
python3 -m experiments.prepare_benchmark
```

Run the benchmark:

```bash
python3 -m experiments.run_benchmark
```

Analyze saved results:

```bash
python3 -m experiments.analyze_benchmark
```

Experiment outputs are saved under `results/benchmark/`.

## Limitations

- The benchmark contains only three research questions.
- The evaluator uses the same underlying model as the generation systems.
- The multi-agent system uses more model calls and tokens than the baseline.
- Retrieved literature does not guarantee that generated claims are supported.
- Proximity scores are recorded but are not yet used to remove similar hypotheses.