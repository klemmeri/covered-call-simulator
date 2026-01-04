from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.bs import covered_call_strike_for_delta

router = APIRouter(prefix="/covered-call", tags=["covered-call"])


class StrikeRequest(BaseModel):
    spot: float
    dte_days: float
    iv: float
    target_delta: float
    r: float = 0.05
    q: float = 0.0

class StrikeLadderRequest(BaseModel):
    spot: float = Field(100, description="SPY spot price")
    dte_days: int = Field(30, description="Days to expiration")
    iv: float = Field(0.25, description="Implied volatility")
    target_deltas: list[float] = Field(
        [0.15, 0.20, 0.25, 0.30],
        description="Target call deltas"
    )
    r: float = Field(0.05, description="Risk-free rate")
    q: float = Field(0.0, description="Dividend yield")


class StrikeLadderResponse(BaseModel):
    symbol: str
    strike_increment: int
    strikes: dict[float, int]



class StrikeResponse(BaseModel):
    strike: float


from fastapi import HTTPException

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

    return StrikeResponse(strike=k)

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

        # ---- ROUND TO NEAREST INTEGER STRIKE ----
        strikes[d] = int(round(k))

    return StrikeLadderResponse(
    symbol="SPY",
    strike_increment=1,
    strikes=strikes,
)


