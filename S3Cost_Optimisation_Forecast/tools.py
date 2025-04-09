from langchain_core.tools import tool
from typing import List,Dict
from load_csv import load_data
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
#from finOps_agent import FinOpsAgent
from schema import GraphState
from prompts import summarize_forecast_prompt
from prophet import Prophet
from langchain_core.messages import AIMessage
from llm_groq import get_groq_llm  # Your LLM setup
from langchain.chains.summarize import load_summarize_chain
from models import TimePeriodInput,LoadDataInput,TotalCostInput,AverageCostInput,AnomalyInput,OptimizationInput,ForecastInput

df = load_data()

# --- Tool Logic ---

#df["Date"] = pd.to_datetime(df["Date"])

@tool(args_schema=TimePeriodInput)
def load_data_for_period(time_period: str) -> dict:
    """
    Parses a time period string and filters the S3 usage DataFrame for rows matching the specified month and year.

    Supports full and abbreviated month names like "March 2025" or "Mar 2025".

    Args:
        time_period (str): A string like "March 2025" or "Nov 2024".

    Returns:
        dict: A dictionary containing filtered S3 usage data rows for the specified period.
    """
    # Try parsing full and abbreviated month formats
    try:
        parsed = pd.to_datetime(time_period, format="%B %Y")
    except ValueError:
        try:
            parsed = pd.to_datetime(time_period, format="%b %Y")
        except ValueError:
            print(f"[load_data_for_period] Invalid time_period format: {time_period}")
            return {"data": []}

    month, year = parsed.month, parsed.year

    df = load_data()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Define the start and end of the month
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    # Filter the data
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    if filtered_df.empty:
        print(f"[load_data_for_period] No data found for: {month}/{year}")
    else:
        print(f"[load_data_for_period] Loaded {len(filtered_df)} rows for: {month}/{year}")

    return {"data": filtered_df.to_dict(orient="records")}

@tool
def compute_total_cost(input: TotalCostInput) -> dict:
    """
    Compute average daily cost for the given data.
    """
    total = sum(row["Cost"] for row in input.data if row["Cost"] is not None)
    return {"total_cost": f"${total:.2f}"}

@tool(args_schema=AverageCostInput)
def compute_average_cost(data: List[Dict]) -> dict:
    """
    Compute average daily cost for the given data.
    """
    if not data:
        return {"avg_cost": "N/A"}

    avg = sum(row.get("Cost", 0) for row in data if row.get("Cost") is not None) / len(data)
    return {"avg_cost": f"${avg:.2f}"}

@tool(args_schema=AnomalyInput)
def detect_anomalies(data: List[Dict]) -> dict:
    """
    Detect anomalies in cost using 2 standard deviation rule.
    Returns a string listing anomaly dates or a message if none found.
    """
    
    if not data:
        return {"anomaly_output": "No data provided."}
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)
    
    if df["Cost"].isnull().all():
        return {"anomaly_output": "No cost data available."}
    
    threshold = df["Cost"].mean() + 2 * df["Cost"].std()
    anomalies = df[df["Cost"] > threshold]

    if anomalies.empty:
        return {"anomaly_output": "No anomalies detected."}
    else:
        dates = ", ".join(anomalies["Date"].astype(str))
        return {"anomaly_output": f"Anomalies detected on: {dates}"}

@tool(args_schema=OptimizationInput)
def suggest_optimizations(data: List[Dict]) -> dict:
    """
    Provide optimization suggestions based on usage patterns and tags.
    """
    if not data:
        return {"optimization": "No data provided."}

    df = pd.DataFrame(data)
    suggestions = []

    # Check for frequent backup usage
    if 'Tag' in df.columns:
        tag_counts = df['Tag'].value_counts(dropna=False)
        if tag_counts.get("backup", 0) > 5:
            suggestions.append("Consider archiving old backups to reduce S3 costs (e.g., move to Glacier).")

    # Check for unusually high cost spikes
    if 'Cost' in df.columns and df["Cost"].notnull().any():
        max_cost = df["Cost"].max()
        if max_cost > 50:
            suggestions.append(f"High cost detected (${max_cost:.2f}). Review for potential misconfigurations or overuse.")

    # Final output
    if suggestions:
        return {"optimization": "💡 Optimization Tips:\n" + "\n".join(f"- {tip}" for tip in suggestions)}
    else:
        return {"optimization": "✅ Usage looks optimized for this time period."}
    
#@tool(args_schema=ForecastInput)

@tool
def cost_forecast(time_period: str) -> dict:
    """
    Predicts next month's S3 usage cost using Prophet and summarizes the insight via LLM.
    """
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    from prophet import Prophet
    from langchain_core.messages import AIMessage

    df = load_data()
    
    # Parse dates and clean data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Cost'])

    # Ensure 'Cost' is numeric
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df = df.dropna(subset=['Cost'])

    # Aggregate by day and rename for Prophet
    daily_costs = df.groupby('Date')['Cost'].sum().reset_index()
    daily_costs = daily_costs.rename(columns={'Date': 'ds', 'Cost': 'y'})

    if len(daily_costs) < 2:
        return {
            "forecast": "Insufficient data to generate a forecast.",
            "raw_prediction": None
        }

    # Train the model
    model = Prophet()
    model.fit(daily_costs)

    # Define forecast period
    today = datetime.today()
    next_month_start = (today.replace(day=1) + relativedelta(months=1)).date()
    next_month_end = (next_month_start + relativedelta(months=1)) - timedelta(days=1)

    # Forecast for the next 31 days
    future = model.make_future_dataframe(periods=31)
    forecast = model.predict(future)

    # Filter forecast for next month
    forecast_period = forecast[
        (forecast['ds'].dt.date >= next_month_start) &
        (forecast['ds'].dt.date <= next_month_end)
    ]
    predicted_cost = round(forecast_period['yhat'].sum(), 2)

    # Calculate historical average of last 3 months
    df['Month'] = df['Date'].dt.to_period('M')
    historical_monthly_cost = df.groupby('Month')['Cost'].sum()
    last_3_months_avg = round(historical_monthly_cost[-3:].mean(), 2)

    delta_percent = ((predicted_cost - last_3_months_avg) / last_3_months_avg) * 100
    trend = "increase" if delta_percent > 0 else "decrease"

    forecast_input = summarize_forecast_prompt.format(
        forecast_cost=predicted_cost,
        trend_direction=trend,
        delta_percent=abs(round(delta_percent, 2))
    )

    llm = get_groq_llm()
    result = llm.invoke(forecast_input)
    forecast_summary = (
        result.content if isinstance(result, AIMessage)
        else str(result)
    )
    print("Forecast range:", next_month_start, "to", next_month_end)
    print("Forecast period rows:", forecast_period.shape[0])
    print("Predicted cost:", predicted_cost)
    return {
        "forecast": forecast_summary,
        "raw_prediction": predicted_cost
    }
