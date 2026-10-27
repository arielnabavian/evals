"""Read a study's record: the goal, every candidate as measured and as decided, how the run ended, the selection and
its test, what it cost, and each setting's effect.

    python examples/optimization/read_record.py studies/quickstart-best-20261026-233747

It reads one search's directory (``study.jsonl``, ``result.json``, ``effects.json``). A study with
``Split(rotate=True)`` keeps one per run, in ``run-1/``, ``run-2/``, ...; give one of those.
"""

# ruff: noqa: T201

import json
import sys
from collections import Counter
from pathlib import Path


def number(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + number(abs(x))


def main(directory: str) -> None:
    record = Path(directory)
    events = [json.loads(line) for line in (record / "study.jsonl").open()]
    header = events[0]
    goal = [f"{o['kind']}({o['metric']})" for o in header["objectives"]]
    limits = [c["name"] for c in header["constraints"]]
    metrics = [o["metric"] for o in header["objectives"]] + [c["metric"] for c in header["constraints"]]
    print(f"goal: {header['combine']['kind']} of {', '.join(goal)}; limits: {limits or 'none'}")
    print(f"events: {dict(Counter(e['type'] for e in events))}")

    print("\ncandidate  role      phase    proposed    measured -> decided (the model's posterior mean)")
    for e in events:
        if e["type"] == "candidate":
            measured = {m: e["estimates"][m]["mean"] for m in metrics if m in e["estimates"]}
            decided = {m: d["mean"] for m, d in (e.get("decided") or {}).get("metrics", {}).items()}
            row = "  ".join(
                f"{m} {number(v)}" + (f" -> {number(decided[m])}" if m in decided else "") for m, v in measured.items()
            )
            proposed = "/".join(e["proposed"]) or "-"
            print(f"{e['number']:9}  {e['role']:8}  {e.get('phase') or '-':7}  {proposed:10}  {row}")
        elif e["type"] == "anchor":
            phase = e["phase"]
            print(f"  anchor: phase A ended ({phase['reason']}) after {phase['steps']} steps; A* = {e['candidate']}")
        elif e["type"] == "finished":
            print(f"finished: {e['reason']}; best {e['best']}" + ("; UNCONFIRMED" if e["unconfirmed"] else ""))

    selected = next(e for e in events if e["type"] == "selected")
    print(
        f"\nselected on {selected['on']} by {selected['by']}: {selected['best']}"
        + (f", the pick {selected['pick']}" if selected.get("pick") is not None else "")
        + (" (UNCONFIRMED: nothing met the limits; this is the closest)" if selected["unconfirmed"] else "")
    )
    result = json.loads((record / "result.json").read_text())
    test = result["test"]
    if test is None:
        print("no test cases: the numbers above are from the cases the search used")
    else:
        for arm in test["selected"]:
            parts = [
                f"{m} {number(arm['means'][m])} vs {number(test['baseline']['means'][m])} "
                f"({signed(d['mean'])}, 90% [{signed(d['ci90'][0])}, {signed(d['ci90'][1])}])"
                for m, d in arm["difference"].items()
            ]
            cases = len(test["cases"])
            print(f"test, candidate {arm['number']} against the baseline on {cases} cases: {'; '.join(parts)}")
    cost = result["cost"]
    print(
        f"cost: {cost['total']['tokens']:,} tokens; by role { ({r: c['tokens'] for r, c in cost['by_role'].items()}) }"
    )

    effects = record / "effects.json"
    if effects.exists():
        print("\neffects (change from the reference level, 90% interval):")
        for metric, fitted in json.loads(effects.read_text())["metrics"].items():
            for effect in fitted["effects"]:
                low, high = (signed(x) for x in effect["ci90"])
                print(f"  {metric}: {effect['key']} = {effect['level']}: {signed(effect['mean'])} [{low}, {high}]")


if __name__ == "__main__":
    main(sys.argv[1])
