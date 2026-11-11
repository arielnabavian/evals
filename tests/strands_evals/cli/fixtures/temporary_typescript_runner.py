import json
import os
import sys

DESCRIPTION = json.loads(os.environ.get("FAKE_RUNNER_DESCRIPTION", "{}"))
LOG = os.environ.get("FAKE_RUNNER_LOG")


def log(request):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


def reply_to(request):
    kind = request["type"]
    if kind == "describe":
        return DESCRIPTION
    if kind == "probe":
        return {"ok": request["values"].get("effort") != "max"}
    effort = request["values"].get("effort", DESCRIPTION.get("effort", "auto"))
    output = f"effort {effort}"
    return {
        "output": output,
        "messages": [
            {"role": "user", "content": [{"text": str(request["input"])}]},
            {"role": "assistant", "content": [{"text": output}]},
        ],
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
        "cycles": 1,
    }


for line in sys.stdin:
    request = json.loads(line)
    log(request)
    if request["type"] == "cancel":
        continue
    sys.stderr.write("noise from the agent module\n")
    sys.stdout.write(json.dumps({"id": request["id"], **reply_to(request)}) + "\n")
    sys.stdout.flush()
