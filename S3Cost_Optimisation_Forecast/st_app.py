import streamlit as st
from langgraph_pipeline import build_graph, graph
from schema import GraphState
import pandas as pd
from finOps_agent import FinOpsAgent
from langchain_core.messages import AIMessage

# --- Page Config ---
st.set_page_config(page_title="FinOps Cloud Cost Advisor")

# --- Title & Intro ---
st.title("FinOps Cloud Cost Advisor")
st.markdown(
    """
    Ask me anything about your **cloud storage (S3)** costs:
    """
)

# --- Initialize FinOps Agent ---
agent = FinOpsAgent()

# --- User Input ---
time_period = st.text_input("Enter the time period (e.g., March 2025):")
if st.button("Analyze"):
    if not time_period:
        st.warning("Please enter a valid time period (e.g., March 2025).")
    else:
        try:
            # Step 1: Load data using FinOpsAgent
            filtered_df = agent.load_data(time_period)
            st.dataframe(filtered_df)

            if filtered_df.empty:
                st.warning("No data available for the selected time period.")
            else:
                # Step 2: Run the LangGraph pipeline
                result = graph.invoke(GraphState({
                    "time_period": time_period,
                    "data": filtered_df
                }))

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Analysis Complete ✅")
                    st.markdown("### Results")
                    st.markdown(f"**Total Cost:**  {result.get('total_cost', 'N/A')}")
                    st.markdown(f"**Average Daily Cost:**  {result.get('avg_cost', 'N/A')}")
                    st.markdown(f"**Anomaly Detection:**  {result.get('anomaly_output', 'N/A')}")
                    st.markdown(f"**Optimization Tips:**  {result.get('optimization', 'N/A')}")

                    # Step 3: Forecast section
                    st.markdown("### Forecast for Next Month")
                    forecast = result.get("forecast", None)
                    prediction = result.get("raw_prediction", "N/A")

                    if isinstance(prediction, (int, float)):
                        st.markdown(f"📈 **Predicted Cost for Next Period:** ${float(prediction):.2f}")
                    elif prediction != "N/A":
                        try:
                            st.markdown(f"📈 **Predicted Cost for Next Period:** ${float(prediction):.2f}")
                        except ValueError:
                            st.warning("⚠️ Predicted cost is not in a valid numeric format.")
                    else:
                        st.warning("Predicted cost is available.")

                    if forecast:
                        st.markdown(f"🧠 **Forecast Summary:** {forecast}")
                    else:
                        st.warning("⚠️ Forecast summary not available.")

                    st.markdown("---")
                    st.caption("This analysis is powered by LangGraph, LangChain, and Groq LLM.")

        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
