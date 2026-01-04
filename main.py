from __future__ import annotations


import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.auth.routes import router as auth_router
from app.core.bs import bs_call_price_greeks, find_strike_for_target_delta
from app.api.covered_call import router as covered_call_router


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app = FastAPI(title="Covered Call Web Simulator API")
app.include_router(covered_call_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRADING_DAYS = 252



# -----------------------------
# Simulation state
# -----------------------------
@dataclass
class Window:
    day: List[int] = field(default_factory=list)
    price: List[float] = field(default_factory=list)
    iv: List[float] = field(default_factory=list)
    strike: List[Optional[float]] = field(default_factory=list)
    cash: List[float] = field(default_factory=list)
    eq_cc: List[float] = field(default_factory=list)
    eq_bh: List[float] = field(default_factory=list)
    call_delta: List[Optional[float]] = field(default_factory=list)
    call_gamma: List[Optional[float]] = field(default_factory=list)
    call_vega: List[Optional[float]] = field(default_factory=list)
    call_theta: List[Optional[float]] = field(default_factory=list)  # per-day
    event: List[str] = field(default_factory=list)
    assigned: List[int] = field(default_factory=list)


@dataclass
class ShortCall:
    strike: float
    dte_rem: int
    open_premium: float  # per share


@dataclass
class RunState:
    run_id: str
    symbol: str
    day: int
    S: float
    iv: float
    r: float
    q: float
    regime: str
    seed: int

    initial_cash: float
    starting_price: float
    chart_window_days: int

    # Strategy knobs persisted
    strategy_target_delta: float = 0.30
    strategy_dte_days: int = 5

    # Shares constrained to 0 or 100 by default
    max_lots: int = 1  # 1 lot = 100 shares

    # State flags
    auto_sell_enabled: bool = True  # whether Step/expiry auto-opens a call if shares exist
    awaiting_user: bool = False     # set True after assignment; user must decide rebuy/open call

    shares_cc: int = 0
    cash_cc: float = 0.0

    shares_bh: int = 0
    cash_bh: float = 0.0

    short_call: Optional[ShortCall] = None
    last_event: str = ""

    basis_per_share: Optional[float] = None

    window: Window = field(default_factory=Window)

    mu_annual: float = 0.09
    sigma_annual: float = 0.18

    iv_mean: float = 0.18
    iv_kappa: float = 0.20
    iv_of_vol: float = 0.06
    iv_ret_coupling: float = 0.60

    rng_state: int = 1


RUNS: Dict[str, RunState] = {}


# -----------------------------
# RNG (simple LCG)
# -----------------------------
def lcg_next(x: int) -> int:
    return (1103515245 * x + 12345) & 0x7FFFFFFF


def rand_uniform01(st: RunState) -> float:
    st.rng_state = lcg_next(st.rng_state)
    return st.rng_state / 0x7FFFFFFF


def randn(st: RunState) -> float:
    u1 = max(1e-12, rand_uniform01(st))
    u2 = max(1e-12, rand_uniform01(st))
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# -----------------------------
# Regime configuration
# -----------------------------
def apply_regime_knobs(st: RunState, regime: str, iv_input: float) -> None:
    st.mu_annual = 0.09
    st.sigma_annual = 0.18
    st.iv_mean = iv_input

    if regime == "uptrend":
        st.mu_annual = 0.09
        st.sigma_annual = 0.14
        st.iv_kappa = 0.25
        st.iv_of_vol = 0.05
        st.iv_ret_coupling = 0.35
    elif regime == "downtrend":
        st.mu_annual = -0.12
        st.sigma_annual = 0.20
        st.iv_kappa = 0.20
        st.iv_of_vol = 0.07
        st.iv_ret_coupling = 0.70
    elif regime == "choppy":
        st.mu_annual = 0.00
        st.sigma_annual = 0.22
        st.iv_kappa = 0.22
        st.iv_of_vol = 0.08
        st.iv_ret_coupling = 0.65
    elif regime == "volatile":
        st.mu_annual = 0.00
        st.sigma_annual = 0.32
        st.iv_kappa = 0.18
        st.iv_of_vol = 0.12
        st.iv_ret_coupling = 0.90
    else:  # mixed
        st.mu_annual = 0.04
        st.sigma_annual = 0.18
        st.iv_kappa = 0.22
        st.iv_of_vol = 0.07
        st.iv_ret_coupling = 0.55


# -----------------------------
# Core mechanics
# -----------------------------
def open_new_call(st: RunState, target_delta: float, dte_days: int) -> str:
    if st.shares_cc <= 0:
        st.short_call = None
        return "NO SHARES | NO CALL"

    dte_days = max(3, int(dte_days))
    T = dte_days / TRADING_DAYS
    K = find_strike_for_target_delta(st.S, T, st.r, st.q, st.iv, target_delta)
    call_price, delta, _, _, _ = bs_call_price_greeks(st.S, K, T, st.r, st.q, st.iv)

    contracts = st.shares_cc // 100
    st.cash_cc += call_price * 100.0 * contracts
    st.short_call = ShortCall(strike=K, dte_rem=dte_days, open_premium=call_price)

    return f"OPEN NEW CALL @ ${K:.0f} (Delta={delta:.2f}, DTE={dte_days})"


def close_call_btc(st: RunState) -> float:
    if not st.short_call or st.shares_cc <= 0:
        st.short_call = None
        return 0.0

    sc = st.short_call
    T = max(0.0, sc.dte_rem / TRADING_DAYS)
    call_price, *_ = bs_call_price_greeks(st.S, sc.strike, T, st.r, st.q, st.iv)

    contracts = st.shares_cc // 100
    cost = call_price * 100.0 * contracts
    st.cash_cc -= cost
    st.short_call = None
    return call_price


def can_rebuy_100(st: RunState) -> bool:
    return st.cash_cc >= (st.S * 100.0)


def rebuy_100_shares(st: RunState) -> str:
    if st.shares_cc > 0:
        return "ALREADY HOLDING SHARES"

    if not can_rebuy_100(st):
        return "INSUFFICIENT CASH TO REBUY 100"

    st.cash_cc -= st.S * 100.0
    st.shares_cc = 100
    st.basis_per_share = st.S
    return f"REBUY 100 @ ${st.S:.2f}"


def compute_snapshot(st: RunState) -> dict:
    call_liab = 0.0
    strike = None
    dte_rem = None
    dlt = gma = vga = thd = None

    if st.short_call and st.shares_cc > 0:
        sc = st.short_call
        strike = sc.strike
        dte_rem = sc.dte_rem
        T = max(0.0, sc.dte_rem / TRADING_DAYS)
        call_price, dlt, gma, vga, th_y = bs_call_price_greeks(st.S, sc.strike, T, st.r, st.q, st.iv)
        contracts = st.shares_cc // 100
        call_liab = call_price * 100.0 * contracts
        thd = th_y / TRADING_DAYS

    eq_cc = st.cash_cc + st.shares_cc * st.S - call_liab
    eq_bh = st.cash_bh + st.shares_bh * st.S

    return {
        "day": st.day,
        "price": st.S,
        "iv": st.iv,
        "shares": st.shares_cc,
        "strike": strike,
        "call_delta": dlt,
        "call_gamma": gma,
        "call_vega": vga,
        "call_theta": thd,
        "cash": st.cash_cc,
        "call_liability": call_liab,
        "dte_remaining": dte_rem,
        "eq_cc": eq_cc,
        "eq_bh": eq_bh,
        "event": st.last_event or "",
        # state for UI
        "awaiting_user": st.awaiting_user,
        "auto_sell_enabled": st.auto_sell_enabled,
        "basis_per_share": st.basis_per_share,
        "can_rebuy_100": can_rebuy_100(st),
        "market_regime": st.regime,
    }


def append_window(st: RunState) -> None:
    snap = compute_snapshot(st)

    st.window.day.append(int(st.day))
    st.window.price.append(float(st.S))
    st.window.iv.append(float(st.iv))
    st.window.cash.append(float(st.cash_cc))
    st.window.eq_cc.append(float(snap["eq_cc"]))
    st.window.eq_bh.append(float(snap["eq_bh"]))
    st.window.event.append(str(st.last_event or ""))

    st.window.strike.append(None if snap["strike"] is None else float(snap["strike"]))
    st.window.assigned.append(1 if "ASSIGNED" in (st.last_event or "") else 0)

    st.window.call_delta.append(None if snap["call_delta"] is None else float(snap["call_delta"]))
    st.window.call_gamma.append(None if snap["call_gamma"] is None else float(snap["call_gamma"]))
    st.window.call_vega.append(None if snap["call_vega"] is None else float(snap["call_vega"]))
    st.window.call_theta.append(None if snap["call_theta"] is None else float(snap["call_theta"]))

    w = st.window
    if len(w.day) > st.chart_window_days:
        for arr_name in (
            "day", "price", "iv", "strike", "cash", "eq_cc", "eq_bh",
            "call_delta", "call_gamma", "call_vega", "call_theta",
            "event", "assigned"
        ):
            arr = getattr(w, arr_name)
            setattr(w, arr_name, arr[-st.chart_window_days:])


def step_one_day(st: RunState) -> None:
    st.last_event = ""

    # 1) simulate return
    mu_d = st.mu_annual / TRADING_DAYS
    sig_d = st.sigma_annual / math.sqrt(TRADING_DAYS)
    z = randn(st)
    ret = mu_d + sig_d * z
    st.S = max(1.0, st.S * math.exp(ret))

    # 2) update IV
    ret_clamped = max(-0.05, min(0.05, ret))
    shock = randn(st)
    iv_prev = st.iv
    mr = st.iv_kappa * (st.iv_mean - iv_prev)
    cluster = st.iv_of_vol * abs(shock)
    coupling = -st.iv_ret_coupling * ret_clamped
    st.iv = max(0.05, min(1.50, iv_prev + mr + cluster + coupling))

    # 3) age option
    if st.short_call:
        st.short_call.dte_rem = max(0, st.short_call.dte_rem - 1)

    # 4) expiration handling
    if st.short_call and st.short_call.dte_rem == 0:
        sc = st.short_call

        if st.S > sc.strike and st.shares_cc > 0:
            # Assignment at expiration
            st.cash_cc += sc.strike * st.shares_cc
            st.shares_cc = 0
            st.short_call = None

            st.awaiting_user = True
            st.auto_sell_enabled = False  # force explicit choice post-assignment
            st.last_event = f"ASSIGNED @ ${sc.strike:.0f} | USER DECISION REQUIRED"

        else:
            # expires worthless
            st.short_call = None
            st.last_event = f"EXPIRE (worthless) @ ${sc.strike:.0f}"

            # only auto-open if enabled and not awaiting user
            if st.auto_sell_enabled and (not st.awaiting_user) and st.shares_cc > 0:
                st.last_event += " | " + open_new_call(st, st.strategy_target_delta, st.strategy_dte_days)

    # 5) if no call exists, auto-open only if enabled AND not awaiting user
    if st.short_call is None and st.shares_cc > 0 and st.auto_sell_enabled and (not st.awaiting_user):
        st.last_event = st.last_event or open_new_call(st, st.strategy_target_delta, st.strategy_dte_days)

    # 6) advance day and append
    st.day += 1
    append_window(st)


# -----------------------------
# API Models
# -----------------------------
class StartRequest(BaseModel):
    symbol: str = "SPY"
    initial_cash: float = 100000.0
    starting_price: float = 480.0
    target_call_delta: float = 0.30
    dte_days: int = 5
    iv: float = 0.18
    r: float = 0.045
    dividend_yield: float = 0.0
    market_regime: str = "mixed"
    seed: int = 1
    chart_window_days: int = 80
    max_lots: int = 1  # enforce 0 or 100 shares by default


class StepRequest(BaseModel):
    run_id: str


class RunNRequest(BaseModel):
    run_id: str
    n: int = 1


class RollRequest(BaseModel):
    run_id: str
    target_delta: float
    dte_days: int


class CloseRequest(BaseModel):
    run_id: str


class ToggleRequest(BaseModel):
    run_id: str


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/start")
def start(req: StartRequest):
    run_id = uuid.uuid4().hex

    st = RunState(
        run_id=run_id,
        symbol=req.symbol,
        day=0,
        S=float(req.starting_price),
        iv=max(0.05, float(req.iv)),
        r=float(req.r),
        q=float(req.dividend_yield),
        regime=req.market_regime,
        seed=int(req.seed),

        initial_cash=float(req.initial_cash),
        starting_price=float(req.starting_price),
        chart_window_days=max(20, int(req.chart_window_days)),

        strategy_target_delta=float(req.target_call_delta),
        strategy_dte_days=max(3, int(req.dte_days)),

        max_lots=max(1, int(req.max_lots)),

        shares_cc=0,
        cash_cc=float(req.initial_cash),

        shares_bh=0,
        cash_bh=float(req.initial_cash),

        rng_state=(int(req.seed) if int(req.seed) > 0 else 1),
    )

    apply_regime_knobs(st, req.market_regime, req.iv)

    # Buy shares for both strategies (lot sizing)
    max_shares = int(st.cash_cc // st.S)
    lot_cap = st.max_lots * 100
    shares = min((max_shares // 100) * 100, lot_cap)

    st.shares_cc = shares
    st.cash_cc -= shares * st.S
    st.basis_per_share = st.S if shares > 0 else None

    st.shares_bh = shares
    st.cash_bh -= shares * st.S

    # Open initial call (auto sell enabled at start)
    st.auto_sell_enabled = True
    st.awaiting_user = False
    if shares > 0:
        st.last_event = open_new_call(st, st.strategy_target_delta, st.strategy_dte_days)

    append_window(st)

    RUNS[run_id] = st
    return {
        "run_id": run_id,
        "state": {"run_id": run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)},
    }


@app.post("/step")
def step(req: StepRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    step_one_day(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/run_n")
def run_n(req: RunNRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    n = max(1, min(2000, int(req.n)))
    for _ in range(n):
        step_one_day(st)
        # if awaiting user, stop advancing (so auto-run UI can pause cleanly)
        if st.awaiting_user:
            break
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.get("/state")
def get_state(run_id: str):
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}


@app.post("/roll_fast")
def roll_fast(req: RollRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    # strategy knobs updated
    st.strategy_target_delta = float(req.target_delta)
    st.strategy_dte_days = max(3, int(req.dte_days))

    btc = close_call_btc(st)
    st.last_event = f"CLOSE CALL (BTC) @ ${btc:.0f} | ROLL -> Delta={st.strategy_target_delta:.2f}, DTE={st.strategy_dte_days}"

    # rolling implies user is in control; clear awaiting flag
    st.awaiting_user = False

    open_new_call(st, st.strategy_target_delta, st.strategy_dte_days)
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/custom_roll")
def custom_roll(req: RollRequest):
    return roll_fast(req)


@app.post("/close_call")
def close_call(req: CloseRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    btc = close_call_btc(st)
    st.auto_sell_enabled = False
    st.last_event = f"CLOSE CALL (BTC) @ ${btc:.0f} | AUTO-SELL DISABLED"
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/open_call")
def open_call(req: RollRequest):
    """
    Uses RollRequest for (run_id, target_delta, dte_days).
    """
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    if st.shares_cc <= 0:
        raise HTTPException(status_code=400, detail="No shares available to cover a call.")

    st.strategy_target_delta = float(req.target_delta)
    st.strategy_dte_days = max(3, int(req.dte_days))

    st.auto_sell_enabled = True
    st.awaiting_user = False

    st.last_event = open_new_call(st, st.strategy_target_delta, st.strategy_dte_days)
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/resume_calls")
def resume_calls(req: ToggleRequest):
    """
    Enables auto-sell (but does NOT open immediately; it will open on next Step if no call exists).
    """
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    st.auto_sell_enabled = True
    st.awaiting_user = False
    st.last_event = "AUTO-SELL ENABLED"
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/rebuy_100")
def rebuy_100(req: ToggleRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    msg = rebuy_100_shares(st)
    st.last_event = msg
    # still awaiting user until they choose open call or stay cash; but after rebuy, we allow open call
    # keep awaiting_user True if they came from assignment flow
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


@app.post("/stay_cash")
def stay_cash(req: ToggleRequest):
    st = RUNS.get(req.run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    st.awaiting_user = False  # user explicitly decided
    st.auto_sell_enabled = False
    st.last_event = "STAY IN CASH | AUTO-SELL DISABLED"
    append_window(st)
    return {"state": {"run_id": st.run_id, "window": st.window.__dict__, "snapshot": compute_snapshot(st)}}


# -----------------------------
# Stats
# -----------------------------
def compute_metrics(eq: List[float], rf_annual: float) -> dict:
    if len(eq) < 2:
        return {
            "start": eq[0] if eq else None,
            "end": eq[-1] if eq else None,
            "total_return": 0.0,
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_days": len(eq) - 1,
        }

    start = eq[0]
    end = eq[-1]
    total_return = (end / start) - 1.0 if start > 0 else 0.0
    n_days = len(eq) - 1
    years = n_days / TRADING_DAYS
    cagr = (end / start) ** (1 / years) - 1.0 if start > 0 and years > 0 else 0.0

    rets = []
    for i in range(1, len(eq)):
        rets.append((eq[i] / eq[i - 1]) - 1.0 if eq[i - 1] > 0 else 0.0)

    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = [r - rf_daily for r in rets]

    mean_ex = sum(excess) / len(excess)
    var = sum((x - mean_ex) ** 2 for x in excess) / max(1, (len(excess) - 1))
    std = math.sqrt(var)

    ann_vol = std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    sharpe = (mean_ex / std) * math.sqrt(TRADING_DAYS) if std > 0 else 0.0

    peak = eq[0]
    mdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0
        if dd < mdd:
            mdd = dd

    wins = sum(1 for r in rets if r > 0)
    win_rate = wins / len(rets)

    return {
        "start": start,
        "end": end,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "n_days": n_days,
    }


@app.get("/stats")
def stats(run_id: str):
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")

    eq_cc = st.window.eq_cc
    eq_bh = st.window.eq_bh
    assignments = sum(st.window.assigned) if st.window.assigned else 0
    n_days = max(0, len(st.window.day) - 1)
    assigned_pct = (assignments / n_days) if n_days > 0 else 0.0

    return {
        "run_id": run_id,
        "n_days": n_days,
        "assignments": assignments,
        "assigned_pct": assigned_pct,
        "rf_annual": st.r,
        "cc": compute_metrics(eq_cc, st.r),
        "bh": compute_metrics(eq_bh, st.r),
    }
