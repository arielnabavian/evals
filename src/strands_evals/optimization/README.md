# Harness optimization

`strands_evals.optimization` tunes a Strands agent's harness against a Strands Evals `Experiment` while the model stays
fixed. You mark which arguments of your agent factory may change. A `Study` searches them, decides with a statistical
model of every goal metric, selects on validation cases, and tests its selection once, on cases nothing before it saw.
Every candidate, trial and token it spends goes into a record you can read afterwards.

Status: a research preview on the `feat/harness-optimization` branch. The API can still change, and one dependency
(BayesBench) is not public yet; [Installing](#installing) says what that means for you.

- [What it is, and why](#what-it-is-and-why)
- [Installing](#installing)
- [Quickstart](#quickstart)
- [The mental model](#the-mental-model)
- [Reading a result and a record](#reading-a-result-and-a-record)
- [Choosing a goal](#choosing-a-goal)
- [Width, depth and concurrency](#width-depth-and-concurrency)
- [Running the benchmark examples](#running-the-benchmark-examples)
- [Known limitations and results so far](#known-limitations-and-results-so-far)
- [Extending it](#extending-it)
- [FAQ and troubleshooting](#faq-and-troubleshooting)
- [Where things are](#where-things-are)

## What it is, and why

An agent is a model inside a harness: the code that builds the model's context, runs its tools, manages its context
window and decides when to call it again. With the model fixed, the harness is what you can change. That means the
system prompt and the harness's own instructions, each tool's description, the thinking effort, which context manager
runs and at what thresholds, and which plugins are on.

Tuning these by hand runs into two problems. The settings have different types and depend on each other: a prompt is
text, an effort level is a category, a truncation threshold is a number, and a summarizing manager's ratio exists only
when that manager is chosen. And every measurement is noisy. An agent on a simulated benchmark passes a task on one run
and fails it on the next, so the configuration that measured best is often the one that got lucky. In our τ² runs,
before the framework selected on held-out cases, the search cases overstated the selected configuration's gain by about
5 points.

Optimizers built around one score, such as GEPA for prompts, propose one kind of variable, keep the best measured
candidate, and often score it on the cases they searched. This framework differs in four ways:

1. Each kind of variable gets a method that can propose it (TPE for numbers and categories, an LLM rewriter for text),
   and structural choices are searched in rounds, so a context manager is compared at its tuned best rather than at its
   defaults.
2. Decisions use a Bayesian model of every goal metric. A candidate that measured lucky is shrunk toward what its
   settings predict, and a limit counts as met only when the model gives it a probability of at least 0.9.
3. The cases that search, select and test are kept apart. The test is a paired difference from the baseline over cases
   with a 90% interval.
4. Every token is counted, the agent's and the optimizer's own, by role and by phase.

## Installing

```bash
git clone https://github.com/jjbuck/evals.git
cd evals
git checkout feat/harness-optimization
python -m venv .venv && . .venv/bin/activate
pip install -e ".[optimization]"
```

Python 3.10 or later. The `optimization` extra adds Optuna (TPE), Jinja2 (the rewriter's templates), NumPy and SciPy.

The study's model (`model.py`) is built on BayesBench, which is not on PyPI and whose repository is not public yet. If
you have access:

```bash
pip install "bayesbench @ git+https://github.com/nonatofabio/bayesbench@fb2d6385d36fcbaabce258b791466a98f263b382"
```

Without it everything still runs, and the study decides with measured means, as `Selection(by="measured")` does. It
tells you once:

```
model: the model could not be fitted (ModuleNotFoundError: No module named 'bayesbench'); deciding with measured means
```

The record's `selected` event then says `"by": "measured"`, a study without a split does not measure its finalists again
before selecting, no `effects.json` is written (the run ends with `effects: not written (ModuleNotFoundError: ...)`),
and the 20 unit tests that need the model skip. The rest of this guide shows output with BayesBench installed; without
it the decided values are the measured ones.

To run the tests (the suite is offline):

```bash
pip install pytest pytest-asyncio
python -m pytest tests/strands_evals/optimization -q
```

## Quickstart

### Offline, in under a minute

```bash
python examples/optimization/quickstart.py
```

`quickstart.py` is a real Strands agent with a scripted model in place of an LLM. It answers sums. It answers correctly
more often at a higher `effort`, with more `steps` in the `stepwise` style, and at a lower `temperature`. Effort and
steps cost it output tokens. Everything else is the framework as it runs on a real model. The part that matters:

```python
@tunable(
    effort=Choice(["low", "medium", "high"], default="low"),
    temperature=Float(0.0, 1.0, default=0.7),
    style=OneOf(component(Concise), component(Stepwise, steps=Int(1, 8, default=2)), labels=["concise", "stepwise"]),
)
def create_agent(effort, temperature, style):
    return Agent(model=ScriptedModel(effort, style, temperature), callback_handler=None)


study = Study(
    agent=create_agent,
    experiment=Experiment(cases=sums(), evaluators=[Correct(name="correct")]),
    objectives=Maximize("correct"),
    repetitions=2,  # each candidate runs every case twice
    split=Split(train=0.5, validation=0.25, test=0.25),  # search on 24 sums, select on 12, test on 12
    max_concurrency=200,
    record=f"studies/quickstart-best-{time.strftime('%Y%m%d-%H%M%S')}",
)
result = study.optimize(iterations=8)
```

A run takes about 3 seconds. Its output, shortened (the scripted model's mistakes are random, so yours will differ):

```
candidate  0 baseline           complete n=24 correct 0.312 [0.151, 0.474]
candidate  1 structure          complete n=24 correct 0.312 [0.163, 0.462]
candidate  2 numbers            complete n=24 correct 0.708 [0.571, 0.846]
...
candidate  5 numbers            complete n=24 correct 0.854 [0.723, 0.985]
...
repeat       structure          = candidate 5: not measured again
...
candidate 28 numbers            complete n=24 correct 0.812 [0.691, 0.934]
...
validated    candidate 0: n=12 correct 0.208 [0, 0.419]
validated    candidate 5: n=12 correct 0.625 [0.43, 0.82]
validated    candidate 14: n=12 correct 0.792 [0.63, 0.954]
validated    candidate 28: n=12 correct 0.792 [0.63, 0.954]
test         candidate 28 (selected) on 12 cases: correct 0.958 vs 0.208 (+0.75 [+0.575, +0.925])
cost         3,183,840 tokens: agent 3,183,840 (search 2,929,920, validation 184,080, test 69,840)

ships candidate 28: {'effort': 'high', 'temperature': 0.2216, 'style': 'stepwise', 'style.stepwise.steps': 6}
record: studies/quickstart-best-20261026-233747
```

Reading it from the top:

- Candidate 0 is the baseline, the agent at its declared defaults. Each `candidate` line gives what proposed it, its
  status, how many cases it ran (`n=24`) and each goal metric's mean over cases with a 95% interval.
- `structure` is a round's opening proposal: TPE choosing the style. `numbers` is a step of TPE over effort, temperature
  and steps under that style. The budget was 8 steps of 4 candidates, 32 in all; a round's opening proposal counts as
  one candidate.
- `repeat` is a proposal that matched a candidate already measured. It is not measured again; its method is told the
  earlier result and proposes again.
- When the search ends, the three best candidates by the model's decided values, and the baseline, run on the 12
  validation cases, and the selection is made from those runs. Candidate 5 was the best on the search cases (0.854) and
  dropped to 0.625 on validation. That gap is the winner's curse, and it is why the selection waits for validation.
- The selection and the baseline then run on the 12 test cases. `+0.75 [+0.575, +0.925]` is the selection minus the
  baseline, paired by case, with a 90% interval.
- `cost` is every token the study spent, by phase. Here only the agent spent any; the scripted model reports 300 input
  tokens a call plus its thinking and steps as output.

### On Amazon Bedrock, about a minute and $0.20

`poem_length.py` learns a system prompt that makes Claude Haiku 4.5 write poems of 40 to 60 words. The prompt is
rewritten by `Reflection`, an agent on Claude Sonnet 5.5 that reads the trials and writes a new version.

```bash
export AWS_PROFILE=your-profile           # credentials with access to both models
export AWS_DEFAULT_REGION=us-east-1       # see the FAQ: AWS_REGION alone may be ignored
BYPASS_TOOL_CONSENT=true python examples/optimization/poem_length.py
```

The rewriter has a shell over the trial transcripts, confined to its own folders, and `BYPASS_TOOL_CONSENT=true` lets it
run commands without asking. Run text studies where model-written shell commands are acceptable, such as a VM or a
container. `POEM_AGENT_MODEL` and `POEM_REWRITER_MODEL` change the two models.

Output, shortened. The rewriter agent prints its own tool calls and a summary of each rewrite:

```
candidate  0 baseline           complete n=6 WordCount 0 [0, 0]

Tool #1: shell
...
Tool #5: submit_optimized_params
I submitted a new system prompt (661 characters, well under the 2000 limit). ...
All 12 baseline conversations scored 0. ... The poems ran from 127 to 192 words. ...
candidate  1 text               complete n=6 WordCount 0.917 [0.702, 1]
...
candidate  2 text               complete n=6 WordCount 0.917 [0.702, 1]
candidate  3 text               complete n=6 WordCount 0.833 [0.562, 1]
candidate  4 text               complete n=6 WordCount 0.75 [0.311, 1]
remeasured   candidate 1: 2 more repetitions
remeasured   candidate 2: 2 more repetitions
remeasured   candidate 3: 2 more repetitions
cost         126,277 tokens: agent 28,474 (search 17,716, remeasurement 10,758); method:text 97,803
candidate 1: reflection rewritten, 12 trials, rewriter tokens 31,410
...
best system prompt:
You write short poems about the topic you are given.

Length rule (strict): the poem must be between 40 and 60 words total. Aim for about 50 words, ...
```

This study has no `Split`, so it selects on the cases it searched. The three finalists are measured twice more first
(`remeasured`), and the model is refitted before it selects. The run took 73 seconds. The agent spent 28k Haiku 4.5
tokens and the rewriter 98k Sonnet 5.5 tokens, 71k of them cache reads. At Anthropic's list prices that is about $0.20;
Bedrock sets its own prices.

## The mental model

```
 @tunable agent factory  ---  the space: TextValue, Int, Float, Choice, OneOf(component(...), ...)
          |
          v
 Search: blocks (a method + the variables it proposes)  ---  schedule: which blocks move at each step
          |                                                   (Nested rounds; PrimaryFirst phases for a cost goal)
          v
 step: Search.width candidates from the incumbent  --->  trials: every train case x repetitions
          ^                                                         |
          |  decided values                                         v
          +-----------  model of every goal metric  <---------  goals: objectives and constraints
                                                                    |  when the search ends
                                                                    v
   finalists + baseline on validation  --->  selection  --->  test, once: selection vs baseline
                                                                    |
                                                                    v
            record: study.jsonl, transcripts/, result.json, test.jsonl, effects.json, the token ledger
```

### The target: an `@tunable` factory

You write an ordinary function that builds an agent and declare its tunable arguments with `@tunable`. Called with no
values, it builds the agent at the declared defaults: the baseline, candidate 0. The study calls it once per trial, so
every trial gets a fresh agent. A candidate's values are flat dotted keys, such as `style.stepwise.steps`.
`Study(agent=...)` also takes a plain function, which it treats as a factory with nothing to tune.

### The space

| Declaration | What it is | Proposed by default by |
|---|---|---|
| `TextValue(default, about=None, max_chars=None)` | free text, such as a system prompt; `about` tells the rewriter what it is, and a value over `max_chars` is never measured | `Reflection()` |
| `Int(low, high, default, step=1, log=False)` | an integer | `TPE()` |
| `Float(low, high, default, step=None, log=False)` | a number | `TPE()` |
| `Choice(choices, default=...)` | a category | `TPE()` |
| `OneOf(component(A, ...), component(B, ...), labels=[...], default=0)` | a structural choice between components, each with its own variables; only the chosen option's variables exist | `TPE(n_startup_trials=2)`, in rounds |
| `tool_descriptions(*tools, max_chars=800)` | one `TextValue` per tool, starting from its own description; the factory gets copies with the candidate's descriptions | `Reflection()` |
| `SkillLibrary(default=None)` | a directory of Agent Skills, each version stored by content in the study's artifact store | `SkillCuration()` |

`component(constructor, **arguments)` builds a fresh object for each trial from fixed values and declared variables.
`Switch(branches, default)` is a structural choice whose branches are mappings of parameters rather than components.

### Blocks and methods

A block is a method and the variables it proposes: `Block(TPE(), window_size)`, `Block(Reflection(), system_prompt,
tools)`. You pass the same Python objects you declared in `@tunable`, so declare each variable once as a variable and
use it in both places (`document_reader.py` does). A component stands for every variable declared in it. Any variable no
block names goes to a default block for its type: `structure` for `OneOf` and `Switch`, `numbers` for `Int`, `Float` and
`Choice`, `text` for `TextValue`, `skills` for `SkillLibrary`.

The methods are:

- `TPE(seed=None, n_startup_trials=4, **options)`: Optuna's multivariate tree-structured Parzen estimator, with one
  Optuna study per combination of the choices outside its block. It is random for its first `n_startup_trials`
  candidates. The default `structure` block uses 2, because it is asked only once a round.
- `Optuna(sampler)`: any Optuna sampler, as in `Optuna(lambda: optuna.samplers.GPSampler())`.
- `Reflection(over_limit_retries=2, **options)`: an LLM agent that rewrites a block's texts. It reads every candidate
  measured so far, ranked for the goal, and the incumbent's trials labelled by success or failure, then writes each text
  it changes in full. The default model is `global.anthropic.claude-sonnet-5-5`; change it with
  `Reflection(model_config={"model_id": ...})`.
- `SkillCuration(**options)`: creates, revises and retires skills in a `SkillLibrary`.
- `Thompson(seed=None)`: Thompson sampling on the study's model, for blocks of choices. It needs BayesBench.

### The schedule

A schedule says which blocks change at each step; every other variable is held at the scope's incumbent, the best
candidate so far.

- `Joint()` moves every block at every step.
- `Alternate(*blocks, iterations=1)` moves one block at a time, in turn.
- `Nested(outer, inner, inner_iterations)` runs rounds. A round opens with one candidate from the outer block (say,
  which context manager), then runs `inner` (default `Joint()`) for `inner_iterations` steps (default 4) under that
  choice, and tells the outer method the round's best. Rounds can nest to any depth.

Without a schedule of your own, the study nests the structural blocks in rounds, outermost first, around `Alternate()`
over the rest (`Search(within=...)` changes the inner schedule); a space with no structural block runs `Joint()`. In the
quickstart that is `Nested(outer=structure, inner=Alternate())`. A goal with a cost objective is wrapped in reward-first
phases, `PrimaryFirst`; see [Choosing a goal](#choosing-a-goal).

`Search(blocks=..., schedule=..., width=4, redraws=3, within=None, phases="auto")` holds all of this, and the study
takes it as `Study(search=...)`.

### Goals

A goal names metrics. A metric is one of these:

- an evaluator's score, under the evaluator's name, in 0 to 1 unless the evaluator sets `score_range = (low, high)`
  (`None` for unbounded);
- the agent's usage: `tokens` (input, output and both cache kinds), `input_tokens`, `output_tokens`,
  `cache_read_tokens`, `cache_write_tokens`, `seconds`, `model_calls`, `removed_messages`;
- a `Derived(name, fn, range=(None, None))` metric computed from each trial's metrics, such as weighted tokens.

`objectives=` takes one `Maximize(metric)` or `Minimize(metric)`, or several wrapped in how they combine: `Pareto(...)`,
`CheapestOnFront(Minimize(cost), Maximize(value), within=0.05)`, `WeightedSum(...)`, `Chebyshev(...)`,
`Lexicographic(...)` or `Independent(...)`. `constraints=` takes `AtLeast(metric, limit)` and `AtMost(metric, limit)`. A
limit can be absolute (the default) or relative to the baseline on the same cases: `AtLeast("reward", -0.05,
compare="difference")` is "at most 5 points below the baseline", and `AtMost("tokens", 0.8, compare="ratio")` is "at
most 80% of the baseline's tokens".

### Evaluation: trials, repetitions and splits

A trial is one case run once. Each candidate runs every case `repetitions` times. A trial ends with a status:

| Status | Meaning | Counts as |
|---|---|---|
| `completed` | the agent finished and the evaluators scored it | its scores |
| `agent_failure` | the agent raised | the worst end of each score's range (`agent_error_score="worst"`) |
| `infrastructure_invalid` | throttling, a dropped connection, or a task that raised `TrialInvalid` | nothing; run again up to `retries` (2) times |
| `verifier_invalid` | every evaluator failed | nothing |

A trial that measured nothing never becomes a zero. Repetitions are averaged within a case first, so the case is the
unit of every estimate and interval. A trial's `task` runs the agent; the default calls
`agent.invoke_async(case.input)`, and a benchmark with its own loop (a simulated user, a world) passes its own.

`Split(train, validation, test)` gives every case one role, as shares (`Split(train=0.5, validation=0.25, test=0.25)`)
or as lists of cases. The search runs on `train`; the finalists and the baseline run on `validation`, where the
selection is made; the selection and the baseline then run once on `test`. A study that has run its test refuses to
search again. `Split(..., rotate=True)` runs one search per test share (4 for a test share of 0.25), side by side, so
every case is tested exactly once, and pools the tests. Without a split, the study searches and selects on every case
and tests nothing.

### Deciding with a model, then selecting

After every step the study fits one Bayesian model per goal metric to every candidate's case means (`model.py`). It has
effects for each option of a choice, for each number's quantile bin and slope, for each rewritten text, for each case,
and a free effect per candidate. A candidate's decided value is the posterior mean of its case average. Decided values
pick each scope's incumbent and are what TPE and the rewriter are told. A constraint holds when the posterior gives it a
probability of at least `feasibility`. Each new candidate is first scored against the previous fit; when those held-out
checks flag the model, the study decides with measured means instead and records why.

`Selection(by="model", rule=Mean(), feasibility=0.9, finalists=3, unmet="closest", also=())` sets how the study decides
and selects. `rule=LowerBound(q=0.1)` decides on a cautious quantile instead of the mean. `finalists` is how many of the
best go on to validation (up to 5 members of a Pareto set). When no finalist, nor the baseline, meets the constraints on
validation, `unmet="closest"` ships the one closest to meeting them and marks it unconfirmed; `unmet="none"` ships
nothing, so the test measures the baseline alone.

### The record and the token ledger

With `record=directory`, the first `optimize()` creates the directory and writes as it goes:

```
directory/
  study.jsonl        one event per line: study (the header), run, scope, candidate, repeat, told, scope_end,
                     model, anchor, remeasured, validated, selected, finished or stopped
  transcripts/       each trial's messages, stored once by content hash
  test.jsonl         the test: test, one tested event per arm, test_end
  result.json        what Study.optimize() returned (Result.to_dict())
  effects.json       each setting's effect on each goal metric, from the model
  artifacts/         what methods wrote: skill-library versions, the rewriter's digests
```

The ledger counts tokens by role and phase. The roles are `agent` (the agent under optimization, every attempt of every
trial, including attempts run again), `method:<block>` (a method's own model calls, such as the rewriter's) and `probes`
(reserved, always 0 so far). The phases are `search`, `remeasurement`, `validation` and `test`. A simulated user or a
judging model belongs to the benchmark and is not counted. `Study.cost` is the ledger so far, and `Result.cost` is the
final one.

## Reading a result and a record

`examples/optimization/read_record.py` prints one search's record. On the quickstart run above:

```bash
python examples/optimization/read_record.py studies/quickstart-best-20261026-233747
```

```
goal: Pareto of Maximize(correct); limits: none
events: {'study': 1, 'run': 1, 'candidate': 33, 'scope': 3, 'model': 9, 'told': 2, 'scope_end': 3, 'repeat': 1, 'validated': 4, 'selected': 1, 'finished': 1}

candidate  role      phase    proposed    measured -> decided (the model's posterior mean)
        0  baseline  -        -           correct 0.312
        1  outer     -        structure   correct 0.312 -> 0.313
        2  step      -        numbers     correct 0.708 -> 0.691
        5  step      -        numbers     correct 0.854 -> 0.835
...
finished: candidates; best [28]

selected on validation by model: [28]
test, candidate 28 against the baseline on 12 cases: correct 0.958 vs 0.208 (+0.75, 90% [+0.575, +0.925])
cost: 3,183,840 tokens; by role {'agent': 3183840, 'probes': 0}

effects (change from the reference level, 90% interval):
  correct: effort = medium: +0.201 [+0.132, +0.269]
  correct: effort = high: +0.321 [+0.255, +0.387]
  correct: temperature = (-inf, 0.242282]: +0.0691 [+0.00593, +0.132]
  correct: style = stepwise: -0.0051 [-0.175, +0.165]
...
```

What each part tells you:

- A single objective is recorded as a `Pareto` of one. `limits` are the constraints, written as the study judges them.
- The events: one `scope` per schedule run and per round, a `model` event per fit, a `told` for each round's outer draw
  once its round ended, `validated` for each finalist and the baseline, and one `selected`.
- `measured -> decided`: the decided value is pulled toward what the candidate's settings predict. Candidate 5's 0.854
  became 0.835, and the 0.625 it scored on validation shows why some pull is warranted.
- `finished` gives the reason the run ended. `candidates` means the budget was spent, `patience` means the best had not
  changed for `patience` steps, and `exhausted` means the schedule had nothing new to propose.
- The effects come from the same model. The scripted model's true effects are +0.20 for medium effort and +0.35 for
  high, both against low, and the intervals cover them. The style's interval is wide because every candidate after the
  baseline was stepwise, so the record says little about the concise style.

The same numbers are in Python, on what `optimize()` returns:

```python
result = study.optimize(iterations=8)
test = result.test  # one search: a Test; with Split(rotate=True): the runs pooled
arm = test.selected[0]
d = arm.difference["correct"]
print(arm.number, arm.params, d.mean, d.ci90, d.cases)
print(result.cost.tokens("agent", "search"), result.cost.total["tokens"])
print(study.best_candidate.params, study.unconfirmed)
```

### A front, and its pick

With two objectives the best is a set, the Pareto front. `python examples/optimization/quickstart.py --goal pareto`
ended like this:

```
phase        primary ended (cap, 5 steps, 20 candidates): A* = candidate 16 (correct 0.729  tokens 1.72e+03); the cost phase keeps correct within 0.02
validated    candidate 0: n=12 correct 0.0417 [0, 0.132]  tokens 470 [470, 470]
validated    candidate 16: n=12 correct 0.875 [0.733, 1]  tokens 1.72e+03 [1.72e+03, 1.72e+03]
validated    candidate 23: n=12 correct 0.583 [0.402, 0.765]  tokens 1.72e+03 [1.72e+03, 1.72e+03]
validated    candidate 21: n=12 correct 0.5 [0.232, 0.768]  tokens 1.96e+03 [1.96e+03, 1.96e+03]
selected     candidate 16 against A* = candidate 16; the seesaw guard dropped 0 (loses 9 of 9), 23 (loses 7 of 9), 21 (loses 6 of 9)
test         candidate 0 (selected) on 12 cases: correct 0.208 vs 0.208 (+0 [+0, +0])  tokens 470 vs 470 (+0 [+0, +0])
test         candidate 16 (selected, the pick; shipped by selection) on 12 cases: correct 0.708 vs 0.208 (+0.5 [+0.229, +0.771])  tokens 1.72e+03 vs 470 (+1.25e+03 [+1.25e+03, +1.25e+03])
```

The validated front is {0, 16}: the baseline is cheapest and candidate 16 is most accurate. Both are tested. One member
ships, the pick (`Run.pick`, the `selected` event's `pick`), here candidate 16. The pick is the cheapest validated
candidate within the margin of the best validated accuracy that also passes the seesaw guard; the next section explains
the guard.

### Unconfirmed

`python examples/optimization/quickstart.py --goal floor --floor 0.95` asks for accuracy the scripted model cannot
reach:

```
selected     candidate 14, UNCONFIRMED: none of the 1 met the constraints on validation; it is the closest (P(met) 0.00, misses by 0.263)
test         candidate 14 (selected, unconfirmed; shipped by selection) on 12 cases: ...  correct 0.458 vs 0.167 (+0.292 [+0.0861, +0.497])
```

Unconfirmed means no candidate the selection chose from met the constraints on validation. The study shipped the one
with the highest probability of meeting them, then the smallest shortfall. The record marks it in `selected`,
`finished`, `test.jsonl` and `result.json` (`Result.unconfirmed` lists the runs). Treat an unconfirmed selection as "the
best we found, below your bar". `Selection(unmet="none")` ships the baseline instead.

## Choosing a goal

The four goals in `quickstart.py`, with what one run of each shipped:

| Goal | Declaration | Use it when | One run shipped |
|---|---|---|---|
| highest reward | `objectives=Maximize("correct")` | quality is all that matters | high effort, 6 steps: 0.958 on test |
| highest reward within a token budget | `objectives=Maximize("correct"), constraints=[AtMost("tokens", 1200)]` | you have a hard cost ceiling per conversation | medium effort, 3 steps, 1,180 tokens: 0.583 |
| fewest tokens above a floor | `objectives=CheapestOnFront(Minimize("tokens"), Maximize("correct")), constraints=[AtLeast("correct", 0.6)]` | you know the quality you need and want it as cheaply as possible | medium effort, 8 steps, 1,780 tokens: 0.667 |
| both, as a front | `objectives=Pareto(Maximize("correct"), Minimize("tokens"))` | you want to see the trade-off before choosing | the front {baseline, high effort concise}; pick: 0.708 at 1,720 tokens |

Prefer a relative limit when the cases that select and test differ from the cases that search. A floor such as
`AtLeast("reward", 0.95)` judged on train cases need not hold on test cases whose baseline is lower, while
`AtLeast("reward", -0.05, compare="difference")` moves with the baseline. `CheapestOnFront` breaks near-ties on cost
(within 5% of the cheapest) toward the higher value.

### Reward first, then cost

A goal with a cost objective (`Pareto(Maximize(...), Minimize(...))`, or `CheapestOnFront`) searches in two phases by
default (`PrimaryFirst`, `Search(phases="auto")`). Told both objectives from the first step, our searches spent their
budget on easy cost cuts, cheap harnesses that fail, and never reached the high-reward end.

1. Phase A searches the primary objective alone, as a max-reward study would. It stops at the first of three things: a
   stall, a floor met with confidence, or its cap of 2/3 of the steps. A stall is 2 steps in a row without a gain, where
   a gain is a candidate that beats the incumbent it came from with probability 0.8 (adjusted for the step's width),
   paired over the search cases; it only counts once the incumbent is within 0.25 of the metric's best. Its best
   candidate is the anchor, A*.
2. Phase B spends the rest of the budget lowering the cost from A*. Its methods are told the cost, and a candidate
   qualifies only if it is no worse than A* on the primary metric, paired over the cases, with probability 0.9 (the
   selection's `feasibility`): within a margin (0.02 of the metric's range by default) for a Pareto goal, or above the
   floor judged as A*'s level plus the paired difference.
3. On validation, the selection is A* or a cheaper candidate that keeps A*'s level and passes the seesaw guard: it may
   lose at most 1 of the cases A* solves on every repetition. The study never ships something worse than A* by more than
   the margin, as validated.

In the Pareto run above, phase A hit its cap after 5 steps (`cap`), and the floor run's phase A ended after 2 steps
because the floor was met (`met`). The guard exists for a candidate that matches A* on the mean while losing cases A*
always solves; in these runs it dropped candidates that lost most of them. The record has an `anchor` event with each
phase-A step's evidence, and every candidate carries its `phase`.

Change the settings with `Search(phases=PrimaryFirst(margin=0.03, cap=0.5, guard=Seesaw(2)))`. `Search(phases=None)`
turns phases off, so every step is told the goal as declared. `Selection(also=("unmet=none", "unanchored", "anchor"))`
records what those other rules would have shipped and tests each one that differs from the selection.

## Width, depth and concurrency

`optimize(iterations=N)` runs N steps one after another; that is the search's depth. Each step proposes
`Search(width=4)` candidates from the same incumbent and measures them together before the model is refitted, so a run
measures about `width` x `iterations` candidates. A round's opening proposal costs one candidate. A step waits for its
slowest trial, so its wall time barely depends on its width, as long as its trials fit in `max_concurrency`.
`optimize(..., patience=k)` stops once the best has not changed for k steps.

`max_concurrency` is the most trials running at once, across all the candidates of a step and all the runs of a
rotation. Its default is 4, which suits a quick check and is far too low for a real study. When a step's trials cannot
all run at once, `optimize` warns:

```
UserWarning: Search(width=4) measures 4 x 48 trials (24 cases x 2 repetitions) in a step: 192 trials at once, but max_concurrency is 100, so steps run in batches and take longer; give max_concurrency=192 to run them at once
```

Set `max_concurrency` to what your model quota can hold. On Bedrock, a request reserves its `max_tokens` against your
tokens-per-minute quota, so bounded `max_tokens` lets more run at once. The τ² and AppWorld examples use 32,768.

What we measured: on τ² telecom, a step of width 4 found about as much as 2 to 3 steps of width 1, and width 8 about 3
to 3.5, at 1.5 to 2.5 times the trials for the same gain. Where the search cases saturate, width and depth made no
difference to test reward that we could detect. On τ³ banking, which kept improving to step 10, 12 steps of width 4 beat
12 of width 1 by 7.6 points [+2.4, +12.9]. In our runs width did not shorten wall time, because a rotated study's four
searches ran as threads of one Python process and the Bedrock pools throttled: a width-4 telecom step took a median 78
to 87 minutes against 9 to 26 at width 1. Running a rotation's searches side by side did help, cutting wall time to
between 1/1.9 and 1/4 of the searches run one after another.

## Running the benchmark examples

Both examples tune `create_harness` (from `strands-harness`) on Claude Haiku 5.5 over three search spaces: `prompt` (the
harness's instructions), `numbers` (thinking effort and the context manager's thresholds) and `full` (both, plus tool
descriptions, the todos plugin, and a choice of context manager, searched in rounds). The goals are `best`, `budget`,
`cheapest`, `joint` (Pareto) and `floor`, on reward and weighted tokens (each token kind weighted by its price relative
to an uncached input token). `--help` lists every option.

A full study is expensive. With 4 candidates a step and 12 steps, our studies spent these totals (raw tokens, then
weighted as input-equivalent tokens):

| Study | Raw tokens | Weighted |
|---|---|---|
| τ² telecom, prompt space, max reward | 6.6 billion | 1.4 billion |
| AppWorld, full space, max reward | 4.0 billion | 1.5 billion |
| τ³ banking, prompt space, max reward | 53 billion | 14 billion |

At Anthropic's list price for Haiku 5.5 ($0.10 per million input tokens, for prompts up to 100k tokens), 1.4 billion
weighted tokens is about $140; Bedrock sets its own prices. The defaults (`--iterations 100 --patience 15`, every task,
τ² at `--concurrency 150`) are a full study. Start with the small runs below.

### τ²-bench

Setup (the τ² data is read from the clone, so install it editable):

```bash
pip install -e ".[optimization]" strands-harness regex rank-bm25 "websockets>=13.0"
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout b7ea9074c1cba482b30687fecdb5c8425fd6f619
pip install -e ./tau2-bench
```

You need Bedrock access to the agent (`TAU2_AGENT_MODEL`, default `global.anthropic.claude-haiku-5-5`), the simulated
user (`TAU2_USER_MODEL`, a LiteLLM model name, default `bedrock/us.anthropic.claude-sonnet-4-6`), and the rewriter
(`TAU2_REFLECTION_MODEL`, default `us.anthropic.claude-sonnet-5-5`). `TAU2_DOMAIN` picks `airline` (the default),
`retail`, `telecom` or `banking_knowledge`. A Sonnet 5.5 user needs `TAU2_USER_TEMPERATURE=none` (see the FAQ).

```bash
# one conversation, the baseline alone
python examples/optimization/tau2_harness.py --baseline --tasks 1 --repetitions 1

# a small study: 20 tasks dealt 10/5/5, 3 steps of 2 candidates
BYPASS_TOOL_CONSENT=true python examples/optimization/tau2_harness.py --space prompt --goal best \
    --tasks 20 --repetitions 1 --iterations 3 --width 2 --concurrency 20
```

The baseline conversation on airline took 152k agent tokens, 82% of them cache reads. The small study took 9 minutes and
10.2 million tokens (0.2 million of them the rewriter's). Its baseline already solved all 10 search tasks, so no rewrite
could beat it and it shipped the baseline: a small run on a saturated domain looks like that. `--rotate` runs one search
per test share so every task is tested once. Records go to `studies/tau2-harness-<domain>-<space>-<goal>-<time>/`.

### AppWorld

`appworld_setup.sh` builds everything under `APPWORLD_HOME` (default `~/appworld`): AppWorld 0.1.3 in its own venv to
serve each task's world, its data (about 200 MB), the AppWorld client (0.2.0, from its repository) installed into your
environment, and AppWorld's code-agent instructions. It needs git-lfs, and `uv` or a Python 3.11 or 3.12 with `venv`.

```bash
pip install -e ".[optimization]" strands-harness
bash examples/optimization/appworld_setup.sh
export APPWORLD_HOME=~/appworld

# two dev tasks, the baseline alone
python examples/optimization/appworld_harness.py --baseline --split dev --n 2 --repetitions 1 --concurrency 2 \
    --out dev-baseline.json

# a small study: 4 tasks from each of train, dev and test_normal
BYPASS_TOOL_CONSENT=true python examples/optimization/appworld_harness.py --space prompt --goal best \
    --n 4 --repetitions 1 --iterations 2 --width 2 --concurrency 8 --out studies
```

A study searches on AppWorld's train tasks, selects on dev and tests on `--test` (`test_normal` by default, or
`test_challenge`). The two baseline tasks took 45 and 57 seconds and about 59k weighted tokens on average. The small
study took 7.5 minutes and 7.0 million tokens. The models are `APPWORLD_AGENT_MODEL` (default
`us.anthropic.claude-haiku-5-5`) and `APPWORLD_REFLECTION_MODEL` (default `us.anthropic.claude-sonnet-5-5`). Keep
`--concurrency` at 100 or below; the FAQ says why.

### Terminal-Bench

The Terminal-Bench 2.1 example is not on this branch. It needs Docker, Harbor and the task images, and it was written
against an earlier version of the framework.

### What listens on the network

Nothing the examples start is reachable from another machine. Each AppWorld trial starts its own world server, and
AppWorld's server binds every interface with no option to change it, so `appworld_workload.py` starts it through a
wrapper that binds 127.0.0.1, checks the address the server reports, and stops any server that reports another address.
Never run AppWorld's own `--docker` mode for these studies: it publishes the port on every interface. The τ² example
starts no server; τ² runs its environments in process. Model calls go out to Bedrock and nothing comes in.

## Known limitations and results so far

Results come from our experiments with Claude Haiku 5.5 as the agent and Sonnet 5.5 as the rewriter, against GEPA and
RRSI at matched budgets, every number on held-out test cases with 90% intervals.

- Most benchmarks saturate. On τ² telecom every max-reward arm reached 98 to 99% (ours from 31-35% to 99%, GEPA and RRSI
  to 98%), and the differences sat below the smallest detectable one (2.2 to 3.0 points). On LOCA-bench no arm gained.
  On Terminal-Bench 2.1 the one difference that excluded zero came partly from a low baseline draw. In most searches,
  90% of the final train gain arrived by the first or second step, so later steps proposed near-copies of their
  incumbent.
- Where there was headroom, the framework won. On AppWorld, in a pre-registered comparison against a GEPA whose
  proposals respected the length limit, our full space scored +6.2 points [+4.5, +8.0] higher, at 1.77 times that GEPA's
  weighted tokens. On τ³ banking our prompt space beat GEPA by +9.0 [+3.4, +14.7] and RRSI by +5.9 [+0.4, +11.4], at
  1.39 and 1.22 times their weighted tokens. On GEPA's own HotpotQA setup it scored about 5.5 points above GEPA re-run
  from GEPA's code, with a stronger rewriter model than GEPA's.
- Its claims held. Over 38 studies, the validation estimate it selected by exceeded the fresh test by a median of 0.9
  points; GEPA's own scores exceeded it by 1.8 to 16.9.
- Its winning harnesses cost more per conversation, and every pre-registered cost-aware comparison failed before the
  reward-first schedule existed. A floor of 0.95 on telecom shipped 79.8% (one fold validated nothing above the floor
  with confidence and shipped its baseline); a Pareto goal on telecom shipped 70.2%, and on AppWorld 73.2%, against
  GEPA's 98.2% and 81.1%; a gain floor on AppWorld beat GEPA on reward but cost 1.40 times its tokens. Reward first,
  then cost (`PrimaryFirst`) and `unmet="closest"` were added in response, and their experiments are still running. No
  result yet supports a cheaper harness at equal reward.
- A study cannot resume. A run that stops keeps its record, but `optimize()` starts a new study; there is no resume from
  a record yet.
- The validation model is not checked the way the search model is. During search, the held-out check moved the incumbent
  to measured means in 26 of 41 searches of width 4 or more.
- There is no command line or YAML yet; the Python API is the only surface.
- BayesBench is not public yet (see [Installing](#installing)).

The module's docstrings mention experiment rounds ("round 3b", "round 3d"); those name the experiments that motivated a
change. Round 3d is the reward-first schedule.

## Extending it

### A new method

A method proposes values for its block (`begin`, which returns a `Draw`) and learns from results (`observe`). A `Draw`
answers the space's `suggest_*` calls while the candidate is built, and its `origin` is recorded with the candidate.
This one draws uniformly at random:

```python
import random

from strands_evals.optimization import Block, Draw, Method, Search


class Uniform(Method):
    """Random search over numbers and categories."""

    kinds = ("float", "int", "choice")  # the variable types it can propose

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def begin(self, block, base, history, structure, study, objective):
        return UniformDraw(self.rng)


class UniformDraw(Draw):
    def __init__(self, rng):
        self.rng, self.origin = rng, {"method": "Uniform"}

    def suggest_float(self, name, low, high, *, step=None, log=False):
        return self.rng.uniform(low, high)

    def suggest_int(self, name, low, high, *, step=1, log=False):
        return self.rng.randrange(low, high + 1, step)

    def suggest_categorical(self, name, choices):
        return self.rng.choice(list(choices))


EFFORT = Choice(["low", "medium", "high"], default="low")
TEMPERATURE = Float(0.0, 1.0, default=0.7)
search = Search(blocks=[Block(Uniform(seed=0), EFFORT, TEMPERATURE)])  # with @tunable(effort=EFFORT, ...)
```

`observe(block, candidate, draw, study, objective)` is called with every result: `draw` is your own draw when you
proposed the candidate, `None` when it is evidence from another block. Set `concurrent = True` when `begin` keeps no
state between calls, so a step's proposals can be made in parallel. A method that calls a model reports that call's
usage in `origin["usage"]`, and the ledger counts it under `method:<block>`. `simulated_document_reader.py` has a
scripted text method (`ScriptedReflection`).

### A custom schedule

A schedule implements `blocks(k, in_scope)`, the block names to move at step k, or `run(scope)` for full control. This
one moves the structure every fourth step and the numbers otherwise:

```python
from strands_evals.optimization import Schedule, Search


class StructureEvery4th(Schedule):
    def blocks(self, k, in_scope):
        return ["structure"] if k % 4 == 0 else [b for b in in_scope if b != "structure"]


search = Search(schedule=StructureEvery4th())
```

A schedule creates candidates only through its scope (`scope.step(blocks)`, `scope.candidate(blocks)`) and rounds only
through `scope.child(...)`, so the budget, the record and what each method is told stay consistent. `Nested`'s rules are
policy objects (`allocation`, `value`, `start`, `order`), so racing outer choices or warm-starting a round is a new
policy rather than a new schedule. `phases.PrimaryFirst` is a schedule built on the same hooks (`aims`, `finalists`,
`decide`).

### Later: control flow as a variable

The framework searches structure, settings and text. The harness's control flow, the program that calls the model and
the tools, is planned as a value of its own (a `Program` declaration) with a method that proposes code. The record
already reserves keys for a program's parameters, per version (`<key>.<version>.<parameter>`), and keeps probe runs
apart from candidates. Nothing searches control flow yet.

## FAQ and troubleshooting

### Calls are throttled

Strands Evals retries throttled calls, and a trial that measured nothing because of the infrastructure is run again
(`Study(retries=2)`) and never scored zero. The benchmark examples also give each agent
`ModelRetryStrategy(max_attempts=20, initial_delay=4, max_delay=240)`, so a throttled call waits for capacity instead of
failing its conversation (long conversations would otherwise fail most). If throttling persists, lower
`max_concurrency`, bound `max_tokens`, or spread studies over accounts and regions: Haiku 5.5 capacity is per account
and geography, and one τ² study at concurrency 1,200 saturated a pool.

### "Too many open files", or credential errors under load

Strands' `BedrockModel` builds a `boto3` session for every model, and a study builds one per trial. At hundreds of
conversations at once that exhausted file descriptors and the instance metadata service. `tau2_harness.py` shares one
session for the process (`OneSession`). Raise the limit with `ulimit -n 65536`. AppWorld trials also watch each server's
output with `select()`, which fails on descriptors above 1023, so keep AppWorld's `--concurrency` at 100 or below.

### Thinking "off" must be sent as disabled

Haiku 5.5 on Bedrock thinks when the request has no `thinking` field, and `create_harness`'s effort `"off"` sends none.
The τ² example sends `{"type": "disabled"}` for `"off"`. Bedrock refuses `block_binding` alongside it.

### "Invalid `signature` in `thinking` block"

A thinking block is bound to the conversation before it, and a context manager that drops or cuts earlier messages
breaks the binding. The examples ask Bedrock to drop such a block (`block_binding: {"prefix_mismatch_behavior":
"drop_block"}` with the beta header `thinking-binding-controls-2026-08-01`).

### Sonnet 5.5 rejects temperature

It returns "`temperature` is deprecated for this model" for any value but its default of 1.0. The rewriter sends 1.0,
which it accepts. For τ²'s simulated user set `TAU2_USER_TEMPERATURE=none` (no temperature is sent);
`TAU2_JUDGE_TEMPERATURE` follows it.

### The model goes to the wrong region

Strands takes the region from the boto3 session (your profile's `region`, or `AWS_DEFAULT_REGION`) and uses `AWS_REGION`
only when the session has none. Set `AWS_DEFAULT_REGION`. An inference profile such as `au.anthropic...` is only valid
from a region of its own geography.

### The rewriter's output runs into the study's lines, and its first submit fails

The rewriter is an agent that prints its tool calls and a summary of each rewrite, and the study's next line can start
on the same line. It often writes its file and submits it in the same turn; the two tool calls run in parallel, the
submit finds no file, and it submits again. Both are harmless.

### "Cannot run the event loop while another loop is running"

A study runs its own event loop, so `optimize()` cannot be called from inside a running one, as in a Jupyter cell. Run
it in a thread: `await asyncio.to_thread(study.optimize, 8)`.

### "... already holds a study; give each study its own record directory"

A record directory holds one study. Use a new directory, as the examples do with a timestamp.

### "this study has measured its test cases; searching again would use them"

A study tests once. Build a new `Study` to search again.

### "every trial of the baseline (the agent at its defaults) failed"

The study stops before searching. The error names the first trial's failure; usually the model ID, its region, or access
to it.

### The concurrency warning

See [Width, depth and concurrency](#width-depth-and-concurrency).

## Where things are

```
src/strands_evals/optimization/
  study.py        Study, Candidate, Scope: building, measuring, deciding, selecting, testing
  space/          @tunable, the domains, component, OneOf, Switch, tool_descriptions, SkillLibrary
  search.py       Search: blocks, schedule, width, redraws, phases
  methods.py      Block, Method, Draw; TPE, Optuna, Thompson, Reflection, SkillCuration
  schedules.py    Joint, Alternate, Nested and its policies
  phases.py       PrimaryFirst, Stall, Seesaw, the anchor and its paired evidence
  projection.py   metrics, Derived, objectives, constraints, the combinations
  model.py        the model of every goal metric
  selection.py    Selection, Mean, LowerBound, closest (unconfirmed)
  split.py        Split, rotation
  trials.py       Trial, TrialStatus, TrialInvalid, running trials
  result.py       Result, Run, Test, Arm, the pooled test
  cost.py         the token ledger
  record.py       the record, and every event's fields
  effects.py      effects fitted to a record
  _reflection/    the rewriter and skill curator (from strands-labs/harness-optimizer)
tests/strands_evals/optimization/   unit tests, offline
examples/optimization/              the examples (see its README)
```
