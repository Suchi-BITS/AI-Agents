from typing import TypedDict, List, Union,Dict, Any, Optional
import pandas as pd

class Row(TypedDict):
    Date: str
    Service: str
    Cost: float
    Tag: Optional[str]

class GraphState(TypedDict, total=False):
    time_period: str
    data: List[Row]  # or use Any if unsure: data: Any
    total_cost: str
    avg_cost: str
    anomaly_output: str
    optimization: str
    forecast: str
