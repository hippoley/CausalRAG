from __future__ import annotations

from typing import Optional

from causalrag.agent.features import RuntimeFeatureFlags
from causalrag.agent.runtime import CausalAgent, create_agent

from .events import ProbeEventSink
from .loop import ObservableCausalAgentLoop


def instrument_agent(
    agent: CausalAgent,
    event_sink: ProbeEventSink,
    *,
    features: Optional[RuntimeFeatureFlags] = None,
) -> CausalAgent:
    """Replace an agent's loop with an observable, ablation-aware loop."""

    base = agent.loop
    agent.loop = ObservableCausalAgentLoop(
        reasoner=base.reasoner,
        tools=base.tools,
        world_model=base.world_model,
        belief_updater=base.belief_updater,
        hypothesis_updater=base.hypothesis_updater,
        goal_evaluator=base.goal_evaluator,
        time_driver=base.time_driver,
        mismatch_policy=base.mismatch_policy,
        event_sink=event_sink,
        features=features,
    )
    return agent


def create_observable_agent(
    *,
    event_sink: ProbeEventSink,
    features: Optional[RuntimeFeatureFlags] = None,
    **kwargs,
) -> CausalAgent:
    """Create a normal CausalRAG agent and attach probe/OTel + feature gates."""

    return instrument_agent(create_agent(**kwargs), event_sink, features=features)
