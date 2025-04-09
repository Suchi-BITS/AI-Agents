from langgraph.graph import StateGraph
from schema import GraphState
from finOps_agent import FinOpsAgent

# Instantiate the agent
agent = FinOpsAgent()

def build_graph():
    workflow = StateGraph(GraphState)

    # Define all nodes as agent method wrappers
    def load_node(state):
        df = agent.load_data(state.time_period)
        return {"data": df}

    def total_node(state):
        total_cost = agent.get_total_cost(state.data)
        return {"total_cost": total_cost}

    def average_node(state):
        avg_cost = agent.get_avg_cost(state.data)
        return {"avg_cost": avg_cost}

    def anomaly_node(state):
        anomalies = agent.detect_anomalies(state.data)
        return {"anomaly_output": anomalies}

    def optimize_node(state):
        optimization = agent.get_optimizations(state.data)
        return {"optimization": optimization}

    def analyze_node(state):
        analysis = agent.analyze_data(state.time_period, state.data)
        return {"analysis": analysis}

    def forecast_node(state):
        forecast_result = agent.forecast_next_month(state.time_period)
        return {
            "forecast": forecast_result.get("forecast", "N/A"),
            "predicted_cost": forecast_result.get("raw_prediction", None)
        }

    # Add nodes to the graph
    workflow.add_node("load", load_node)
    workflow.add_node("total", total_node)
    workflow.add_node("average", average_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("anomaly", anomaly_node)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("forecast_node", forecast_node)

    # Define edges
    workflow.set_entry_point("load")
    workflow.add_edge("load", "total")
    workflow.add_edge("total", "average")
    workflow.add_edge("average", "analyze")
    workflow.add_edge("analyze", "anomaly")
    workflow.add_edge("anomaly", "optimize")
    workflow.add_edge("optimize", "forecast_node")

    # Build the graph
    return workflow.compile()

# Instantiate the graph
graph = build_graph()
