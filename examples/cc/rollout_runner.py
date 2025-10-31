from cc_agent import CodingAgent

import agentlightning as agl

tracer = agl.OtelTracer()
runner = agl.LitAgentRunner(tracer)
store = agl.InMemoryLightningStore()

with runner.run_context(agent=CodingAgent(), store=store):
    rollout = await runner.step()
