"""The document-reader examples (examples/optimization): the workload, the steps, and the simulated studies run end to
end (a scripted model; no network)."""

import json
import re
import sys
from pathlib import Path

from strands_evals.types.evaluation import EvaluationData

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "optimization"))

import document_reader_steps as steps  # noqa: E402
import document_reader_workload as workload  # noqa: E402
import simulated_document_reader as simulated  # noqa: E402


def test_pages_are_chained_and_hold_one_digit_each():
    page = workload.open_document._tool_func("atlas")
    digits = []
    for n in range(1, workload.PAGES + 1):
        assert f"page {n} of {workload.PAGES}" in page
        digits.append(re.search(r"Digit \d+ of the code is (\d)", page).group(1))
        key = re.search(r"The key to page \d+ is ([0-9a-f]{8})", page)
        if n < workload.PAGES:
            page = workload.read_page._tool_func(key.group(1))
        else:
            assert key is None and "last page" in page
    assert "".join(digits) == workload.DOCUMENTS["atlas"]
    assert workload.read_page._tool_func("00000000") == "No page has that key."


def test_answer_scores_the_ticket_and_each_digit_in_place():
    case = workload.cases()[0]
    ticket, code = case.expected_output.split(": ")

    def score(reply):
        data = EvaluationData(input=case.input, actual_output=reply, expected_output=case.expected_output)
        return workload.Answer().evaluate(data)[0].score

    assert score(f"{ticket}: {code}") == 1.0
    assert score(f"{code}") == 10 / 11  # ticket lost
    wrong = code[:-1] + str((int(code[-1]) + 1) % 10)
    assert score(f"{ticket}: {wrong}") == 10 / 11
    assert score("I could not finish.") == 0.0


def test_each_step_adds_one_idea(tmp_path):
    built = {n: steps.study(n, str(tmp_path / str(n)), log=None) for n in steps.STEPS}
    blocks = {n: sorted(s.blocks) for n, s in built.items()}
    assert blocks == {
        1: ["numbers"],
        2: ["numbers"],
        3: ["text"],
        4: ["numbers", "text"],
        5: ["numbers", "structure", "text"],
        6: ["numbers", "structure", "text"],
        7: ["numbers"],
        8: ["numbers", "text"],
    }
    goals = {n: ([o.metric for o in s.projection.objectives], len(s.projection.constraints)) for n, s in built.items()}
    assert goals == {
        1: (["answer"], 0),
        2: (["answer", "tokens"], 0),
        3: (["answer"], 0),
        4: (["answer", "tokens"], 0),
        5: (["answer", "tokens"], 0),
        6: (["answer", "tokens"], 2),
        7: (["tokens"], 1),  # step 2's space, with a quality floor in place of the answer objective
        8: (["answer"], 1),  # step 4's space, with a token budget in place of the token objective
    }
    assert set(built[2].params) == set(built[1].params) | {"pin_first"}
    assert set(built[6].params) - set(built[5].params) == {"tools.open_document", "tools.read_page"}
    assert set(built[7].params) == set(built[2].params) and set(built[8].params) == set(built[4].params)
    # a goal with a cost objective (answer and tokens) searches the answer first, then the tokens (round 3d): its
    # schedule is PrimaryFirst around the one it ran before
    kinds = [type(s.schedule).__name__ for s in built.values()]
    assert kinds == ["Joint", "PrimaryFirst", "Joint", "PrimaryFirst", "PrimaryFirst", "PrimaryFirst", "Joint", "Joint"]
    inner = [type(getattr(s.schedule, "inner", s.schedule)).__name__ for s in built.values()]
    assert inner == ["Joint"] * 4 + ["Nested"] * 2 + ["Joint"] * 2
    assert not any((tmp_path / str(n)).exists() for n in steps.STEPS)  # building a study writes nothing


def test_the_simulated_study_runs_end_to_end_and_writes_its_record(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "make_model", steps.make_model)  # restored afterwards: the simulator replaces it
    study = simulated.study(6, str(tmp_path / "sim"), log=None)
    study.optimize(iterations=3)

    width = study.search.width  # each of the 3 steps proposes up to this many (identical proposals measured once)
    assert 4 <= len(study.candidates) <= 1 + 3 * width
    assert study.baseline.role == "baseline"
    events = [json.loads(line) for line in (tmp_path / "sim" / "study.jsonl").read_text().splitlines()]
    remeasured = {e["number"] for e in events if e["type"] == "remeasured"}  # finalists, measured again at the end
    for c in study.candidates:
        if c.status != "invalid":
            assert len(c.trials) == 2 * len(workload.cases()) * (2 if c.number in remeasured else 1)
    header = events[0]
    assert header["type"] == "study" and header["format"] == "strands_evals.optimization"
    assert header["repetitions"] == 2 and header["cases"] == list(workload.DOCUMENTS)
    assert header["blocks"]["text"]["method"] == {"kind": "ScriptedReflection"}
    candidates = [e for e in events if e["type"] == "candidate"]
    assert [c["number"] for c in candidates] == list(range(len(study.candidates)))
    assert {t["case"] for t in candidates[0]["trials"]} == set(workload.DOCUMENTS)
    assert events[-1]["type"] == "finished"
    texts = [c["origin"]["text"] for c in candidates if "text" in c["origin"]]
    assert texts and all(o["method"] == "ScriptedReflection" and "from_candidate" in o for o in texts)


def test_every_simulated_step_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "make_model", steps.make_model)
    for n in steps.STEPS:
        study = simulated.study(n, str(tmp_path / str(n)), log=None)
        study.optimize(iterations=2)
        assert all(c.status == "complete" for c in study.candidates)
        end = json.loads((tmp_path / str(n) / "study.jsonl").read_text().splitlines()[-1])
        # a scripted rewrite of a candidate that lost nothing changes nothing: the run can end with nothing new
        assert len(study.candidates) <= 1 + 2 * study.search.width and end["reason"] in ("candidates", "exhausted")
