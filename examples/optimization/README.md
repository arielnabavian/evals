# Optimization examples

Studies with `strands_evals.optimization`. The guide, with the concepts, the quickstart and what each example's output
means, is [src/strands_evals/optimization/README.md](../../src/strands_evals/optimization/README.md). Run every
command from the repository root.

| File | What it is | Needs |
|---|---|---|
| `quickstart.py` | A whole study in seconds: a real Strands agent with a scripted model, four goals (`--goal best`, `budget`, `floor`, `pareto`). | nothing beyond `[optimization]` |
| `read_record.py` | Prints a study's record: every candidate as measured and as decided, the selection, its test, the cost, the effects. | nothing |
| `poem_length.py` | The smallest real study: Reflection rewrites a system prompt until the poems are 40 to 60 words. About a minute. | Bedrock: Claude Haiku 4.5, Claude Sonnet 5.5 |
| `document_reader.py` | The annotated study: every argument of `Study`, `Search` and the space commented, on an agent that reads a chained document under a conversation manager. | Bedrock: Claude Haiku 4.5, Claude Sonnet 5.5 |
| `document_reader_steps.py` | The document-reader study in six steps, each adding one idea (one number, a category and tokens, text, two blocks, rounds over a structural choice, constraints). | Bedrock |
| `simulated_document_reader.py` | The six steps with a scripted model and a scripted rewrite. No network. The unit tests run it. | nothing |
| `document_reader_workload.py` | The document-reader tools, cases and evaluator. Not a study. | |
| `tau2_harness.py`, `tau2_workload.py` | `create_harness` as a τ²-bench agent: three search spaces by five goals, with rotation. | τ²-bench, strands-harness, Bedrock |
| `appworld_harness.py`, `appworld_workload.py`, `appworld_setup.sh` | `create_harness` as an AppWorld code agent, and the script that sets AppWorld up. | AppWorld, strands-harness, git-lfs, Bedrock |

Records go to `studies/` (ignored by git). Each study needs a directory of its own.

Studies whose search space holds text run Reflection, whose agent has a shell over the trial transcripts, confined to
its own folders. `BYPASS_TOOL_CONSENT=true` lets that shell run without asking: run those studies where model-written
shell commands are acceptable, such as a VM or a container.
