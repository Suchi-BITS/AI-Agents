from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from llm_groq import llm
from tools import toolkit

#tools = [add, multiply, find_max]

agent = initialize_agent(
    tools=toolkit,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,handle_parsing_errors=True,**{
        "max_iterations": 2,        # ⬅️ Increase this
        "max_execution_time": 60     # ⬅️ Seconds (1 minute)
    }
)

