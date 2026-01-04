# app/api/sim_state.py
from __future__ import annotations

from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field


EventType = Literal[
    "BUY_SHARES",
    "SELL_CALL",
    "ASSIGNED",
    "BUYBACK_SHARES",
    "EXPIRE_CALL",
    "NO_ACTION",
    "CLOSE_CALL",
]


class Event(BaseModel):
    day: int = Field(..., description="Simulation day index")
    type: EventType
    details: Dict[str, Any] = Field(default_factory=dict)


class ShortCall(BaseModel):
    strike: float = Field(..., description="Call strike (SPY uses $1 increments)")
    dte_days: int = Field(..., description="Days to expiration at open")
    delta_target: Optional[float] = Field(None, description="Target delta used to choose strike")
    credit_per_share: float = Field(..., description="Option premium received PER SHARE")
    opened_day: int = Field(..., description="Day the call was opened")


class SimState(BaseModel):
    spot: float
    day: int = 0
    shares: int = 100
    cash: float = 0.0
    short_call: Optional[ShortCall] = None
    events: List[Event] = Field(default_factory=list)


class InitializeRequest(BaseModel):
    spot: float = Field(100, description="Initial SPY spot price")
    start_cash: float = Field(0.0, description="Starting cash balance (optional)")


class InitializeResponse(SimState):
    pass

class AdvanceDayRequest(BaseModel):
    spot: float = Field(..., description="New spot price for the next day")


class AdvanceDayResponse(SimState):
    pass

class RebuySharesRequest(BaseModel):
    price: float = Field(..., description="Price to rebuy 100 shares")


class RebuySharesResponse(SimState):
    pass

class SellCallRequest(BaseModel):
    strike: float = Field(..., description="Call strike (SPY uses $1 increments)")
    dte_days: int = Field(..., description="Days to expiration for the short call")
    delta_target: float = Field(0.30, description="Target delta used to select strike (optional)")
    credit_per_share: float = Field(..., description="Premium received per share")


class SellCallResponse(SimState):
    pass
