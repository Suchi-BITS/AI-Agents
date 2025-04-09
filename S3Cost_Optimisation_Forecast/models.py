# models.py

from pydantic import BaseModel,Field
from typing import List, Dict
from datetime import datetime
import calendar

class TimePeriodInput(BaseModel):
    time_period: str
    
class LoadDataInput(BaseModel):
    month: int
    year: int
    
class TotalCostInput(BaseModel):
    data: List[Dict]

class AverageCostInput(BaseModel):
    data: List[Dict]

class AnomalyInput(BaseModel):
    data: List[Dict]

class OptimizationInput(BaseModel):
    data: List[Dict]

class ForecastInput(BaseModel):
    time_period: str 