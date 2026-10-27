"""examples/optimization/appworld_harness.py's goals (round 3b): the floor goal's absolute floor, or gain over the
baseline."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("strands_harness")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "optimization"))
import appworld_harness as ah  # noqa: E402

from strands_evals.optimization import AtLeast, CheapestOnFront  # noqa: E402


def test_the_floor_goal_takes_an_absolute_floor_or_a_gain_over_the_baseline():
    gain = ah.goals("floor", floor_gain=0.072)
    assert isinstance(gain["objectives"], CheapestOnFront) and gain["objectives"].within == ah.NEAR
    assert gain["constraints"] == [AtLeast("reward", 0.072, compare="difference")]
    assert ah.goals("floor", floor=0.95)["constraints"] == [AtLeast("reward", 0.95)]
    for wrong in ({}, {"floor": 0.95, "floor_gain": 0.072}):
        with pytest.raises(ValueError, match="one of them"):
            ah.goals("floor", **wrong)
    assert ah.goals("cheapest")["constraints"] == [AtLeast("reward", -0.05, compare="difference")]  # round 3's


def test_the_cost_goals_search_reward_first_unless_the_schedule_is_joint():
    from strands_evals.optimization import PrimaryFirst

    phased = ah.phases("joint", "reward-first", 0.02, 1, 2 / 3)
    assert isinstance(phased, PrimaryFirst) and phased.margin == 0.02 and phased.guard.k == 1 and phased.cap == 2 / 3
    assert isinstance(ah.phases("floor", "reward-first", None, None, 0.5), PrimaryFirst)
    assert ah.phases("floor", "reward-first", None, None, 0.5).guard is None  # --guard -1: no guard
    assert ah.phases("joint", "joint", 0.02, 1, 2 / 3) is None  # round 3 and 3b's schedule
    assert ah.phases("best", "reward-first", 0.02, 1, 2 / 3) is None  # max reward: no phases
    with pytest.raises(ValueError, match="--schedule"):
        ah.phases("joint", "together", 0.02, 1, 2 / 3)
