from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.bs import covered_call_strike_for_delta
from app.api.sim_state import InitializeRequest, InitializeResponse, SimState, Event  # keep if used elsewhere

router = APIRouter(prefix="/covered-call", tags=["covered-call"])


# ----------------------------
# Models
# ----------------------------

class StrikeRequest(BaseModel):
    spot: float = Field(100, description="SPY spot price")
    dte_days: int = Field(30, description="Days to expiration")
    iv: float = Field(0.25, description="Implied volatility")
    target_delta: float = Field(0.30, description="Target call delta (0.01 to 0.99)")
    r: float = Field(0.05, description="Risk-free rate")
    q: float = Field(0.0, description="Dividend yield")


class StrikeLadderRequest(BaseModel):
    spot: float = Field(100, description="SPY spot price")
    dte_days: int = Field(30, description="Days to expiration")
    iv: float = Field(0.25, description="Implied volatility")
    target_deltas: list[float] = Field(
        default_factory=lambda: [0.15, 0.20, 0.25, 0.30],
        description="Target call deltas",
    )
    r: float = Field(0.05, description="Risk-free rate")
    q: float = Field(0.0, description="Dividend yield")


class StrikeResponse(BaseModel):
    strike: float


class StrikeLadderResponse(BaseModel):
    symbol: str
    strike_increment: int
    strikes: dict[float, int]


# ----------------------------
# Endpoints
# ----------------------------

@router.post("/strike-for-delta", response_model=StrikeResponse)
def strike_for_delta(req: StrikeRequest) -> StrikeResponse:
    # ---- Validation ----
    if req.spot <= 0:
        raise HTTPException(status_code=400, detail="spot must be > 0")

    if req.dte_days <= 0:
        raise HTTPException(status_code=400, detail="dte_days must be > 0")

    if req.iv <= 0:
        raise HTTPException(status_code=400, detail="iv must be > 0")

    if not (0.01 <= req.target_delta <= 0.99):
        raise HTTPException(
            status_code=400,
            detail="target_delta must be between 0.01 and 0.99",
        )

    # ---- Compute strike ----
    k = covered_call_strike_for_delta(
        spot=req.spot,
        dte_days=req.dte_days,
        iv=req.iv,
        target_delta=req.target_delta,
        r=req.r,
        q=req.q,
    )

    return StrikeResponse(strike=int(round(k)))



@router.post("/strike-ladder", response_model=StrikeLadderResponse)
def strike_ladder(req: StrikeLadderRequest) -> StrikeLadderResponse:
    # ---- Validation ----
    if req.spot <= 0:
        raise HTTPException(status_code=400, detail="spot must be > 0")

    if req.dte_days <= 0:
        raise HTTPException(status_code=400, detail="dte_days must be > 0")

    if req.iv <= 0:
        raise HTTPException(status_code=400, detail="iv must be > 0")

    if not req.target_deltas:
        raise HTTPException(status_code=400, detail="target_deltas must not be empty")

    strikes: dict[float, int] = {}

    for d in req.target_deltas:
        if not (0.01 <= d <= 0.99):
            raise HTTPException(
                status_code=400,
                detail=f"target_delta {d} must be between 0.01 and 0.99",
            )

        k = covered_call_strike_for_delta(
            spot=req.spot,
            dte_days=req.dte_days,
            iv=req.iv,
            target_delta=d,
            r=req.r,
            q=req.q,
        )

        # ---- ROUND TO NEAREST $1 STRIKE (SPY convention) ----
        strikes[d] = int(round(k))

    return StrikeLadderResponse(
        symbol="SPY",
        strike_increment=1,
        strikes=strikes,
    )
