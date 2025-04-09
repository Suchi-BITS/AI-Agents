from langgraph.graph import StateGraph
from schema import GraphState
from calculator_agent import agent

# This node function will be used in the graph
def call_agent(state: GraphState) -> GraphState:
    query = state["query"]
    a, b, c = state["a"], state["b"], state["c"]
    input_text = f"{query} the numbers {a}, {b}, and {c}"
    result = agent.run(input_text)
    return {"result": result, **state}

# Build the graph pipeline
workflow = StateGraph(GraphState)
workflow.add_node("agent", call_agent)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

# ✅ This is the compiled graph object used in `st_app.py`
graph = workflow.compile()
