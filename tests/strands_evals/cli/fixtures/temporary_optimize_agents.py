from strands import Agent

from strands_evals.optimization import OneOf, tunable
from tests.strands_evals.optimization.scripted_model import ScriptedModel


@tunable(style=OneOf("short", "long", default="short"))
def create_agent(style):
    return Agent(model=ScriptedModel([f"style {style}"]), callback_handler=None)


def plain_factory():
    return Agent(model=ScriptedModel(["plain"]), callback_handler=None)
