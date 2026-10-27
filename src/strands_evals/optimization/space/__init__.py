"""Search spaces: an ordinary agent factory whose arguments are declared tunable.

    @tunable(
        system_prompt=TextValue(default="You answer support tickets concisely."),
        conversation_manager=OneOf(
            component(SlidingWindowConversationManager, window_size=Int(10, 80, default=40)),
            component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
        ),
    )
    def create_agent(system_prompt, conversation_manager): ...

Values are flat dotted keys (``conversation_manager.sliding_window.window_size``); called with none, the factory builds
the agent at its defaults.
"""

from .component import Component, OneOf, component
from .domains import Choice, Domain, Float, Int, SkillLibrary, TextValue
from .factory import TunableFactory, Values, tunable
from .space import ParamInfo, SpaceError
from .switch import Switch, candidate_key
from .tools import tool_descriptions

__all__ = [
    "Choice",
    "Component",
    "Domain",
    "Float",
    "Int",
    "OneOf",
    "ParamInfo",
    "SkillLibrary",
    "SpaceError",
    "Switch",
    "TextValue",
    "TunableFactory",
    "Values",
    "candidate_key",
    "component",
    "tool_descriptions",
    "tunable",
]
