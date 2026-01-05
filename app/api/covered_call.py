from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.api.sim_state import AdvanceDayRequest, AdvanceDayResponse
from app.api.sim_state import RebuySharesRequest, RebuySharesResponse
from app.api.sim_state import SellCallRequest, SellCallResponse, ShortCall



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
@router.post("/advance-day", response_model=AdvanceDayResponse)
def advance_day(state: SimState, req: AdvanceDayRequest) -> AdvanceDayResponse:
    # Update spot and day
    state.day += 1
    state.spot = req.spot

    # If no short call, nothing to age/expire
    if state.short_call is None:
        state.events.append(Event(day=state.day, type="NO_ACTION", details={"spot": state.spot}))
        return AdvanceDayResponse(**state.model_dump())

    # Age the option by one day
    state.short_call.dte_days -= 1

    # If not expired yet, just record and return
    if state.short_call.dte_days > 0:
        state.events.append(
            Event(
                day=state.day,
                type="NO_ACTION",
                details={"spot": state.spot, "call_dte": state.short_call.dte_days},
            )
        )
        return AdvanceDayResponse(**state.model_dump())

    # Expiration day: decide assignment vs expire worthless
    strike = state.short_call.strike

    if state.spot >= strike:
        # Assigned: shares called away, keep premium already in cash (handled when sold)
        state.shares = 0
        state.short_call = None
        state.events.append(
            Event(day=state.day, type="ASSIGNED", details={"spot": state.spot, "strike": strike})
        )
    else:
        # Expires worthless
        state.short_call = None
        state.events.append(
            Event(day=state.day, type="EXPIRE_CALL", details={"spot": state.spot, "strike": strike})
        )

    return AdvanceDayResponse(**state.model_dump())

@router.post("/initialize", response_model=InitializeResponse)
def initialize(req: InitializeRequest) -> InitializeResponse:
    state = SimState(
        spot=req.spot,
        day=0,
        shares=100,
        cash=req.start_cash - 100.0 * req.spot,
        short_call=None,
        events=[Event(day=0, type="BUY_SHARES", details={"shares": 100, "price": req.spot})],
    )
    return InitializeResponse(**state.model_dump())

@router.post("/apply-event", response_model=SimState)
def apply_event(state: SimState, event: Event) -> SimState:
    # advance day if needed
    state.day = event.day

    if event.type == "SELL_CALL":
        state.short_call = event.details

    if event.type == "BUY_SHARES":
        state.shares += event.details["shares"]
        state.cash -= event.details["shares"] * event.details["price"]

    if event.type == "SELL_CALL":
        # credit premium: per-share × 100 shares
        credit = event.details["credit_per_share"] * state.shares
        state.cash += credit

        state.short_call = {
         "strike": event.details["strike"],
         "dte_days": event.details["dte_days"],
         "delta_target": event.details.get("delta_target"),
         "credit_per_share": event.details["credit_per_share"],
         "opened_day": event.day,
    }
    

    state.events.append(event)
    return state

@router.post("/rebuy-shares", response_model=RebuySharesResponse)
def rebuy_shares(state: SimState, req: RebuySharesRequest) -> RebuySharesResponse:
    if state.shares != 0:
        raise HTTPException(status_code=400, detail="Shares already owned (shares must be 0 to rebuy)")

    cost = 100.0 * req.price
    if state.cash < cost:
        raise HTTPException(status_code=400, detail="Insufficient cash to rebuy 100 shares")

    state.shares = 100
    state.cash -= cost

    state.events.append(
        Event(day=state.day, type="BUYBACK_SHARES", details={"shares": 100, "price": req.price})
    )

    return RebuySharesResponse(**state.model_dump())

@router.post("/sell-call", response_model=SellCallResponse)
def sell_call(state: SimState, req: SellCallRequest) -> SellCallResponse:
    if state.shares != 100:
        raise HTTPException(status_code=400, detail="Must own 100 shares to sell a covered call")

    if state.short_call is not None:
        raise HTTPException(status_code=400, detail="Short call already open")

    if req.dte_days <= 0:
        raise HTTPException(status_code=400, detail="dte_days must be > 0")

    # Round strike to nearest $1 (SPY convention)
    strike_int = int(round(req.strike))

    # Credit premium (per share × 100 shares)
    state.cash += float(req.credit_per_share) * 100.0

    state.short_call = ShortCall(
        strike=float(strike_int),
        dte_days=int(req.dte_days),
        delta_target=float(req.delta_target),
        credit_per_share=float(req.credit_per_share),
        opened_day=int(state.day),
    )

    state.events.append(
        Event(
            day=state.day,
            type="SELL_CALL",
            details={
                "strike": strike_int,
                "dte_days": int(req.dte_days),
                "delta_target": float(req.delta_target),
                "credit_per_share": float(req.credit_per_share),
            },
        )
    )

    return SellCallResponse(**state.model_dump())

