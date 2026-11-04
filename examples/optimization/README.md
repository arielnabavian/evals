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
| `toy_two_objectives.py` | Two objectives with closed-form answers (a choice and a number; a bump in quality, a straight-line cost), searched by the real framework with no model calls, for every goal and schedule; each shipped candidate is checked against the true answer. | nothing beyond `[optimization]` |
| `programs_live.py` | One `ProgramProposer` call: its agent writes a `ConversationManager`, which is stored by hash, loaded under the restrictions and built into an agent. No study. | Bedrock: Claude Sonnet 5.5 |
| `alfworld_harness.py`, `alfworld_workload.py`, `alfworld_game.py` | A small open-weight model (Qwen3.5-2B) on an OpenAI-compatible endpoint playing ALFWorld's text games: its system prompt, tool description, thinking, temperature, hints and conversation manager; `--macro` adds a program-valued macro tool written by `ProgramProposer`, `--skills` a skill library. | ALFWorld and its data, `strands-agents[openai]`, an OpenAI-compatible server such as vLLM, Bedrock for the rewriter and the proposer |
| `alfworld_programs.py` | The ALFWorld macro: the `Macro` and `HelperMacro` interfaces a version implements, its checks, the tool that runs it, and how the proposer reads a trial. Not a study. | |

Records go to `studies/` (ignored by git). Each study needs a directory of its own.

Studies whose search space holds text run Reflection, whose agent has a shell over the trial transcripts, confined to
its own folders. `BYPASS_TOOL_CONSENT=true` lets that shell run without asking: run those studies where model-written
shell commands are acceptable, such as a VM or a container. Studies with a `Program` (the ALFWorld macro) run
model-written code in the study's process, under restrictions that are not a sandbox: run them there too.

The ALFWorld example's model server is yours to start. Bind it to 127.0.0.1 (vLLM listens on every interface unless
given `--host`), and reach a server on another machine through an SSH tunnel rather than a published port; the
guide's [ALFWorld section](../../src/strands_evals/optimization/README.md#alfworld-on-an-openai-compatible-endpoint)
has the commands.
