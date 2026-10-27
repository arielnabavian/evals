"""The records the reflection optimizers read: one attempt (``Rollout``) and its score (``Reward``).

Ported from strands-labs/harness-optimizer, ``strands_harness_optimizer/datamodels.py``. Only ``Reward`` and
``Rollout`` are ported; the agent-creator and agent-invoker protocols there are not used by the optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reward:
    """The score of one rollout, with whatever the scorer said about it (``metadata``, e.g. its reasons)."""

    reward: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Rollout:
    """One attempt: the data sample it ran on, the agent's messages, and optional metrics and metadata."""

    data_sample: dict
    messages: list[dict]
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
