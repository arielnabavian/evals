"""One ALFWorld game in a process of its own: alfworld_workload.py starts this script for each episode.

A game is a PDDL problem that TextWorld runs with a planner library (fast-downward), and each game loads its own copy
of that library (34 MB) and keeps the game's state in it. The parsers TextWorld shares in a process are not safe for two
threads, and a process crashed when about 60 games were loaded at once, so each episode has a process: its commands
arrive as JSON lines on stdin, and each reply (the state after the command) leaves as a JSON line on stdout.

    python alfworld_game.py GAME_FILE

Requests: {"command": "go to desk 1"}, {"walkthrough": true} (the game's expert plan, in the names the game uses), or
{"close": true}. Replies: {"feedback", "won", "admissible"}, or {"walkthrough": [...]}, or {"error": "..."}.
"""

import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")  # the importing of numpy would otherwise start a thread per core

import textworld  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos  # noqa: E402


def state_of(state) -> dict:
    return {
        "feedback": state.feedback,
        "won": bool(state.get("won")),
        "admissible": list(state.admissible_commands or []),
    }


def named(env, walkthrough: list[str]) -> list[str]:
    """The game's walkthrough with each object's id replaced by the name the game's text gives it."""
    infos = env._entity_infos
    names = {i: info.name for i, info in infos.items() if info.name and info.name != i}
    pattern = re.compile("|".join(re.escape(i) for i in sorted(names, key=len, reverse=True)))
    return [pattern.sub(lambda m: names[m.group(0)], c) for c in walkthrough]


def main(game_file: str) -> None:
    out = sys.stdout  # the replies; whatever else prints goes to stderr
    sys.stdout = sys.stderr
    try:
        env = textworld.start(
            game_file, textworld.EnvInfos(won=True, admissible_commands=True), wrappers=[AlfredDemangler(), AlfredInfos]
        )
        reply = state_of(env.reset())
    except Exception as exc:  # noqa: BLE001 - reported to the episode
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    out.write(json.dumps(reply) + "\n")
    out.flush()
    if "error" in reply:
        return
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            return
        try:
            if "walkthrough" in request:
                reply = {"walkthrough": named(env, json.load(open(game_file)).get("walkthrough") or [])}
            else:
                state, _, _ = env.step(request["command"])
                reply = state_of(state)
        except Exception as exc:  # noqa: BLE001 - reported to the episode
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main(sys.argv[1])
