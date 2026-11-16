from strands import Agent

from strands_evals.optimization import Choice, tunable
from tests.strands_evals.optimization.helpers import Says


@tunable(style=Choice(["short", "long"], default="short"))
def create_agent(style):
    return Agent(model=Says(text=f"style {style}"), callback_handler=None)


def plain_factory():
    return Agent(model=Says(text="plain"), callback_handler=None)
