from langchain.chains import LLMChain
from llm_groq import get_groq_llm
from prompts import analyze_prompt, summarize_forecast_prompt
from tools import (
    load_data_for_period,
    compute_total_cost,
    compute_average_cost,
    detect_anomalies,
    suggest_optimizations,
    cost_forecast
)
import pandas as pd

class FinOpsAgent:
    def __init__(self):
        self.llm = get_groq_llm()
        self.analyze_chain = LLMChain(llm=self.llm, prompt=analyze_prompt)
        self.forecast_chain = LLMChain(llm=self.llm, prompt=summarize_forecast_prompt)

    def load_data(self, time_period: str) -> pd.DataFrame:
        result = load_data_for_period.invoke({"time_period": time_period})
        return pd.DataFrame(result["data"])

    def get_total_cost(self, data: pd.DataFrame) -> str:
        return compute_total_cost.invoke({"data": data.to_dict(orient="records")})["total_cost"]

    def get_avg_cost(self, data: pd.DataFrame) -> str:
        return compute_average_cost.invoke({"data": data.to_dict(orient="records")})["avg_cost"]

    def detect_anomalies(self, data: pd.DataFrame) -> str:
        return detect_anomalies.invoke({"data": data.to_dict(orient="records")})["anomaly_output"]

    def get_optimizations(self, data: pd.DataFrame) -> str:
        return suggest_optimizations.invoke({"data": data.to_dict(orient="records")})["optimization"]

    def analyze_data(self, time_period: str, data: pd.DataFrame) -> str:
        data_str = data[["Date", "Cost", "Tag"]].to_string(index=False)
        return self.analyze_chain.run(time_period=time_period, data=data_str)

    def forecast_next_month(self, time_period: str) -> dict:
        return cost_forecast.invoke(time_period=time_period)
