"""One live ProgramProposer call (Bedrock, Sonnet 5.5): its agent writes a ConversationManager, which is stored by hash,
loaded under the restrictions, and built into a Strands agent. No study, no benchmark, no measurement of quality: a
look at what the proposer writes and how a version is stored and loaded, for about one model call.

    AWS_REGION=us-west-2 python examples/optimization/programs_live.py [out-dir]

The version is stored under out-dir/artifacts (default: a temporary directory). It is loaded and built into an agent
that is never invoked; only its apply_management runs, on 40 messages.
"""

import sys
import tempfile
from pathlib import Path

from strands import Agent
from strands.agent.conversation_manager import ConversationManager

from strands_evals.optimization.programs import Program, ProgramProposer, load, store

SEED = """from strands.agent.conversation_manager import ConversationManager


class KeepLast(ConversationManager):
    def __init__(self):
        super().__init__()
        self.keep = 8

    def apply_management(self, agent, **kwargs):
        excess = len(agent.messages) - self.keep
        if excess > 0:
            del agent.messages[:excess]
            self.removed_message_count += excess

    def reduce_context(self, agent, e=None, **kwargs):
        self.keep = max(2, self.keep // 2)
        self.apply_management(agent)
"""

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
    manager = Program(ConversationManager, seeds=[SEED], about="how a support agent trims its conversation history")
    version = manager.arguments["version"]
    source, origin = ProgramProposer().write(version, SEED, ["- the seed: not measured yet"])
    print("origin:", origin)  # noqa: T201
    assert source is not None
    value = store(source, out / "artifacts", version.manifest(parent="seed"))
    print("value:", value, "->", out / "artifacts" / value / "program.py")  # noqa: T201
    print("----- source -----\n" + source + "------------------")  # noqa: T201
    try:
        cls = load(source, ConversationManager, version.imports)
    except Exception as exc:  # noqa: BLE001
        print(f"LOADED: no ({type(exc).__name__}: {exc})")  # noqa: T201
        raise SystemExit(1) from exc
    agent = Agent(conversation_manager=cls(), callback_handler=None)  # built, never invoked
    agent.messages.extend(
        {"role": "user" if i % 2 == 0 else "assistant", "content": [{"text": f"message {i}"}]} for i in range(40)
    )
    agent.conversation_manager.apply_management(agent)
    print(f"LOADED: yes ({cls.__name__}); built into Agent; apply_management: 40 -> {len(agent.messages)} messages")  # noqa: T201
