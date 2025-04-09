from langchain.prompts import PromptTemplate

analyze_prompt = PromptTemplate.from_template("""
You are a cloud cost analyst. Analyze the following S3 usage data for the period: {time_period}.

Data:
{data}

Return:
- Total cost
- Average daily cost
- Any anomalies detected (spikes or dips)
- Optimization tips to reduce cost
- Cost Forecast
""")

# Prompt to summarize forecast for the upcoming period
summarize_forecast_prompt = PromptTemplate.from_template("""
The predicted S3 cost for next month is ${forecast_cost}.
This is a {trend_direction} of {delta_percent}% compared to the recent average.

Summarize this in a FinOps advisor tone, highlighting potential causes like backup uploads, traffic spikes, or seasonal changes.
Provide an actionable recommendation if necessary.
""")