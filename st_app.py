import streamlit as st
#from langgraph_pipeline import graph
from langgraph_pipeline import graph  # Ensure this imports your compiled LangGraph

# --- Page Config ---
st.set_page_config(page_title="My Calculator")

# --- Title & Intro ---
st.title("My Calculator")
st.markdown(
    """
    Perform Calculation
    """
)
a = st.number_input("Enter number 1", step=1)
b = st.number_input("Enter number 2", step=1)
c = st.number_input("Enter number 3", step=1)
query = st.text_input("What do you want to do? (e.g. add, multiply, max)")
# --- User Input ---
if st.button("Run"):
    # ✅ This is the state (dictionary) that LangGraph expects
    state = {
        "a": int(a),
        "b": int(b),
        "c": int(c),
        "query": query
    }

    result = graph.invoke(state)

    # ✅ Display the result returned by LangGraph
    st.success(f"Result: {result['result']}")