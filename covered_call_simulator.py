"""
Interactive Covered Call Simulator (Synthetic SPY, 1 path per run)
------------------------------------------------------------------
BETA v0.2 – Substack Edition

Key features
- One synthetic SPY price path per run (no downloads)
- User makes decisions each day:
    [K]eep, [C]lose, [R]oll FAST, [RR]oll with prompt, [Q]uit
- Regime selection at start: choppy / uptrend / downtrend / volatile / mixed
- Plot shows:
    - Price (left axis)
    - Strike (red horizontal line)
    - Cash (right axis, dashed)
- Running totals in plot title:
    - Strategy total equity (Eq)
    - Benchmark buy & hold equity (BH)
    - Realized P/L (Real)
    - Unrealized P/L (Unrl)
    - Cash
- Outputs one Excel workbook (general use, cross-platform):
    ./output/CC_Simulation_Output.xlsx
  with three sheets:
    - EquityCurve
    - TradeLog
    - DecisionLog
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  # set BEFORE pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.stats import norm


# ============================================================
# OUTPUT DIRECTORY (GENERAL, CROSS-PLATFORM)  [OPTION 2]
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "CC_Simulation_Output.xlsx"


# ============================================================
# Formatting helpers
# ============================================================
def money0(x: float) -> str:
    return f"${x:,.0f}"


def dollars_tick(x, pos=None) -> str:
    return f"${x:,.0f}"


# ============================================================
# Black–Scholes (European call, no dividends)
# ============================================================
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_target_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    K_low: float = 0.5,
    K_high: float = 2.0,
    tol: float = 1e-5,
    max_iter: int = 160,
) -> float:
    """
    Find strike K such that call delta approx target_delta using bisection.
    For covered calls we typically want OTM calls: target_delta ~ 0.2-0.4.
    """
    target_delta = min(max(target_delta, 1e-4), 0.9999)
    lo, hi = K_low * S, K_high * S
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d = bs_call_delta(S, mid, T, r, sigma)
        if abs(d - target_delta) < tol:
            return mid
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return mid


# ============================================================
# Synthetic SPY path with regimes (no downloads)
# ============================================================
@dataclass
class PathConfig:
    S0: float = 450.0
    years: int = 3
    trading_days: int = 252

    mu: float = 0.08
    vol_low: float = 0.14
    vol_high: float = 0.28
    p_switch: float = 0.02
    seed: int = 7

    use_mean_reversion: bool = False
    mr_strength: float = 0.15
    mr_anchor: float = 450.0


def choose_regime() -> Tuple[str, PathConfig]:
    print("\n--- Choose Market Regime (Substack mode) ---")
    print("1) Choppy / Mean-Reverting")
    print("2) Uptrending")
    print("3) Downtrending")
    print("4) Volatile / Whipsaw")
    print("5) Random / Mixed (switching)")
    raw = input("Select 1-5 [1]: ").strip() or "1"

    cfg = PathConfig()

    if raw == "1":
        name = "Choppy / Mean-Reverting"
        cfg.mu = 0.03
        cfg.vol_low = 0.12
        cfg.vol_high = 0.20
        cfg.p_switch = 0.04
        cfg.use_mean_reversion = True
        cfg.mr_strength = 0.25
        cfg.mr_anchor = cfg.S0
    elif raw == "2":
        name = "Uptrending"
        cfg.mu = 0.12
        cfg.vol_low = 0.12
        cfg.vol_high = 0.22
        cfg.p_switch = 0.02
    elif raw == "3":
        name = "Downtrending"
        cfg.mu = -0.10
        cfg.vol_low = 0.14
        cfg.vol_high = 0.30
        cfg.p_switch = 0.03
    elif raw == "4":
        name = "Volatile / Whipsaw"
        cfg.mu = 0.02
        cfg.vol_low = 0.18
        cfg.vol_high = 0.40
        cfg.p_switch = 0.07
    elif raw == "5":
        name = "Random / Mixed (switching)"
        cfg.mu = 0.07
        cfg.vol_low = 0.12
        cfg.vol_high = 0.35
        cfg.p_switch = 0.08
    else:
        name = "Choppy / Mean-Reverting"
        cfg.mu = 0.03
        cfg.vol_low = 0.12
        cfg.vol_high = 0.20
        cfg.p_switch = 0.04
        cfg.use_mean_reversion = True
        cfg.mr_strength = 0.25
        cfg.mr_anchor = cfg.S0

    seed = input(f"Random seed (integer) [{cfg.seed}]: ").strip()
    if seed:
        try:
            cfg.seed = int(seed)
        except ValueError:
            print("Seed invalid; keeping default.")
    return name, cfg


def simulate_synthetic_spy(cfg: PathConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.years * cfg.trading_days
    dt = 1.0 / cfg.trading_days

    S = np.zeros(n, dtype=float)
    iv = np.zeros(n, dtype=float)

    S[0] = cfg.S0
    high = False
    iv[0] = cfg.vol_low

    for t in range(1, n):
        if rng.random() < cfg.p_switch:
            high = not high
        sigma = cfg.vol_high if high else cfg.vol_low
        iv[t] = sigma

        z = rng.normal()

        mr_term = 0.0
        if cfg.use_mean_reversion:
            mr_term = cfg.mr_strength * math.log(cfg.mr_anchor / max(S[t - 1], 1e-9))

        S[t] = S[t - 1] * math.exp(
            (cfg.mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z + mr_term * dt
        )

    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"date": dates, "S": S, "iv": iv})


# ============================================================
# Strategy config + option position
# ============================================================
@dataclass
class StrategyConfig:
    shares: int = 100
    default_delta: float = 0.30
    default_dte_days: int = 14
    r: float = 0.04
    trading_days: int = 252
    commission_option: float = 1.00
    commission_stock: float = 0.00
    auto_rebuy_after_assignment: bool = True
    plot_lookback_days: int = 120


@dataclass
class OptionPos:
    is_open: bool = False
    K: float = float("nan")
    entry_idx: int = -1
    expiry_idx: int = -1
    delta_target: float = float("nan")
    dte_days: int = 0


# ============================================================
# Input helpers
# ============================================================
def ask_float(prompt: str, default: float, lo: float = None, hi: float = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if lo is not None and val < lo:
            print(f"Must be >= {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"Must be <= {hi}.")
            continue
        return val


def ask_int(prompt: str, default: int, lo: int = None, hi: int = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            val = int(default)
        else:
            try:
                val = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if lo is not None and val < lo:
            print(f"Must be >= {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"Must be <= {hi}.")
            continue
        return val


def prompt_new_call_params(prev_delta: float, prev_dte: int) -> Tuple[float, int]:
    print("\nOpen NEW CALL parameters (press Enter to keep current):")
    delta = ask_float("  Target delta", prev_delta, lo=0.01, hi=0.99)
    dte = ask_int("  DTE days", prev_dte, lo=1, hi=365)
    return delta, dte


# ============================================================
# Interactive simulator
# ============================================================
def run_interactive(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    starting_cash: float
) -> Dict[str, Optional[pd.DataFrame]]:

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---- Main account state ----
    cash = float(starting_cash)
    shares = 0
    basis = 0.0  # average cost per share for "lot" (simple model)

    realized_pl = 0.0
    pos = OptionPos()

    # Logs
    trades: List[Dict] = []
    equity_rows: List[Dict] = []
    decisions: List[Dict] = []

    # ---- Benchmark: Buy & Hold ----
    S0 = float(df.loc[0, "S"])
    bench_cash = float(starting_cash)
    bench_shares = 0
    bench_basis = 0.0

    bench_cost = cfg.shares * S0
    if bench_cash >= bench_cost:
        bench_shares = cfg.shares
        bench_basis = S0
        bench_cash -= bench_cost
    else:
        bench_shares = 0
        bench_basis = 0.0

    # ---- Strategy: must afford 100 shares to begin ----
    cost = cfg.shares * S0 + cfg.commission_stock
    if cash < cost:
        shortfall = cost - cash
        print("\n*** NOT ENOUGH CASH TO START STRATEGY ***")
        print(f"Need {money0(cost)} to buy {cfg.shares} shares at {S0:.2f}.")
        print(f"You have {money0(cash)} (short by {money0(shortfall)}).")
        return {
            "equity": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "decisions": pd.DataFrame(),
            "figure": None
        }

    shares = cfg.shares
    basis = S0
    cash -= cost
    trades.append({
        "date": df.loc[0, "date"],
        "action": "BUY_SHARES",
        "reason": "INIT",
        "S": S0,
        "shares": shares,
        "cash_flow": -cost,
        "basis": basis
    })

    initial_equity = cash + shares * S0  # baseline after initial buy
    last_delta = cfg.default_delta
    last_dte = cfg.default_dte_days

    def option_mark(i: int) -> Dict[str, float]:
        if not pos.is_open:
            return {"value": 0.0, "delta": 0.0, "days_left": 0}
        S = float(df.loc[i, "S"])
        iv = float(df.loc[i, "iv"])
        days_left = max(pos.expiry_idx - i, 0)
        T = days_left / cfg.trading_days
        value_total = bs_call_price(S, pos.K, T, cfg.r, iv) * cfg.shares
        delta = bs_call_delta(S, pos.K, T, cfg.r, iv)
        return {"value": float(value_total), "delta": float(delta), "days_left": int(days_left)}

    def open_call(i: int, reason: str, prompt_params: bool) -> None:
        nonlocal cash, realized_pl, pos, last_delta, last_dte
        if shares <= 0:
            return

        if prompt_params:
            last_delta, last_dte = prompt_new_call_params(last_delta, last_dte)

        S = float(df.loc[i, "S"])
        iv = float(df.loc[i, "iv"])
        T = last_dte / cfg.trading_days

        K = find_strike_for_target_delta(S, T, cfg.r, iv, last_delta)
        prem = bs_call_price(S, K, T, cfg.r, iv)
        credit = prem * cfg.shares

        cash += credit - cfg.commission_option
        realized_pl += credit - cfg.commission_option

        pos.is_open = True
        pos.K = float(K)
        pos.entry_idx = i
        pos.expiry_idx = min(i + last_dte, len(df) - 1)
        pos.delta_target = float(last_delta)
        pos.dte_days = int(last_dte)

        trades.append({
            "date": df.loc[i, "date"],
            "action": "SELL_CALL",
            "reason": reason,
            "S": S,
            "iv": iv,
            "K": float(K),
            "dte": int(last_dte),
            "delta_target": float(last_delta),
            "premium_per_share": float(prem),
            "cash_flow": float(credit - cfg.commission_option),
        })

    def close_call(i: int, reason: str) -> None:
        nonlocal cash, realized_pl, pos
        if not pos.is_open:
            return
        mark = option_mark(i)
        value_total = mark["value"]

        cash -= value_total + cfg.commission_option
        realized_pl -= (value_total + cfg.commission_option)

        trades.append({
            "date": df.loc[i, "date"],
            "action": "BUY_TO_CLOSE",
            "reason": reason,
            "S": float(df.loc[i, "S"]),
            "iv": float(df.loc[i, "iv"]),
            "K": float(pos.K),
            "days_left": int(mark["days_left"]),
            "value_per_share": float(value_total / cfg.shares),
            "cash_flow": float(-(value_total + cfg.commission_option)),
            "delta_target": float(pos.delta_target),
            "dte": int(pos.dte_days),
        })
        pos = OptionPos()

    def assign_and_rebuy_if_needed(i: int, strike: float) -> None:
        nonlocal cash, shares, basis, realized_pl

        stock_pl = cfg.shares * (strike - basis) - cfg.commission_stock
        realized_pl += stock_pl

        cash += cfg.shares * strike - cfg.commission_stock
        trades.append({
            "date": df.loc[i, "date"],
            "action": "ASSIGNED_SELL_SHARES",
            "reason": "EXPIRE_ITM_ASSIGNMENT",
            "S": float(df.loc[i, "S"]),
            "K": float(strike),
            "cash_flow": float(cfg.shares * strike - cfg.commission_stock),
            "stock_pl_vs_basis": float(stock_pl),
        })

        shares = 0

        if cfg.auto_rebuy_after_assignment and i < len(df) - 1:
            S_next = float(df.loc[i + 1, "S"])
            cost2 = cfg.shares * S_next + cfg.commission_stock
            if cash >= cost2:
                shares = cfg.shares
                basis = S_next
                cash -= cost2
                trades.append({
                    "date": df.loc[i + 1, "date"],
                    "action": "REBUY_SHARES",
                    "reason": "AUTO_REBUY_AFTER_ASSIGNMENT",
                    "S": S_next,
                    "shares": shares,
                    "cash_flow": float(-cost2),
                    "new_basis": float(basis),
                })
            else:
                shortfall = cost2 - cash
                print("\n*** NOT ENOUGH CASH TO REBUY AFTER ASSIGNMENT ***")
                print(f"Need {money0(cost2)} to rebuy {cfg.shares} shares; cash={money0(cash)}; short={money0(shortfall)}.")
                trades.append({
                    "date": df.loc[i + 1, "date"],
                    "action": "REBUY_FAILED",
                    "reason": "INSUFFICIENT_CASH_AFTER_ASSIGNMENT",
                    "S": S_next,
                    "required_cash": float(cost2),
                    "cash_available": float(cash),
                    "shortfall": float(shortfall),
                })

    # ---- Plot setup ----
    plt.ion()
    fig, ax_price = plt.subplots(figsize=(12, 5))
    ax_cash = ax_price.twinx()
    ax_cash.yaxis.set_label_position("right")
    ax_cash.yaxis.tick_right()
    plt.show(block=False)

    def redraw(i: int, total_equity: float, bench_equity: float, unrealized_pl: float,
               dte_left: int, delta_now: float) -> None:

        ax_price.clear()
        ax_cash.clear()

        ax_cash.yaxis.set_major_formatter(FuncFormatter(dollars_tick))
        ax_cash.yaxis.set_label_position("right")
        ax_cash.yaxis.tick_right()

        start = max(0, i - cfg.plot_lookback_days)
        x = df.loc[start:i, "date"]
        y_price = df.loc[start:i, "S"]

        ax_price.plot(x, y_price, linewidth=2, label="Price")
        ax_price.scatter([df.loc[i, "date"]], [df.loc[i, "S"]], s=40, label="Today")

        if pos.is_open:
            ax_price.axhline(pos.K, color="red", linewidth=2, label="Strike")

        y_cash = [row["cash"] for row in equity_rows[start:i + 1]]
        ax_cash.plot(x, y_cash, linewidth=2, linestyle="--", label="Cash")

        title = (
            "Eq=" + money0(total_equity) + "  |  "
            "BH=" + money0(bench_equity) + "  |  "
            "Real=" + money0(realized_pl) + "  |  "
            "Unrl=" + money0(unrealized_pl) + "  |  "
            "Cash=" + money0(cash)
        )
        if pos.is_open:
            title += f"  |  DTE_left={dte_left}"
        ax_price.set_title(title)

        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price")
        ax_cash.set_ylabel("Cash", rotation=90, labelpad=12)

        # Extra right margin for the info box (so it doesn't cover the plot)
        fig.subplots_adjust(left=0.07, right=0.70, top=0.90, bottom=0.16)

        if pos.is_open:
            info = (
                f"DTE_left: {dte_left}\n"
                f"Δ now: {delta_now:.2f}\n"
                f"Strike: {pos.K:.2f}\n"
                f"Δ target: {pos.delta_target:.2f}\n"
                f"DTE (opened): {pos.dte_days}"
            )
        else:
            info = "No short call open"

        ax_price.text(
            1.10, 0.98, info,
            transform=ax_price.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.95),
            fontsize=10,
            clip_on=False
        )

        l1, lab1 = ax_price.get_legend_handles_labels()
        l2, lab2 = ax_cash.get_legend_handles_labels()
        ax_price.legend(l1 + l2, lab1 + lab2, loc="upper left")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # Open initial call (prompt)
    open_call(0, reason="INIT", prompt_params=True)

    # ---- Main loop ----
    for i in range(len(df)):
        date = df.loc[i, "date"]
        S = float(df.loc[i, "S"])
        iv = float(df.loc[i, "iv"])

        mark = option_mark(i)
        opt_liab = mark["value"]
        delta_now = mark["delta"] if pos.is_open else float("nan")
        dte_left = mark["days_left"] if pos.is_open else 0

        stock_value = shares * S
        total_equity = cash + stock_value - opt_liab
        unrealized_pl = total_equity - (initial_equity + realized_pl)

        bench_equity = bench_cash + bench_shares * S

        equity_rows.append({
            "date": date,
            "S": S,
            "iv": iv,
            "cash": cash,
            "shares": shares,
            "basis": basis,
            "stock_value": stock_value,
            "short_call_K": pos.K if pos.is_open else np.nan,
            "call_days_left": dte_left,
            "call_delta_now": delta_now,
            "delta_target": pos.delta_target if pos.is_open else np.nan,
            "call_dte_opened": pos.dte_days if pos.is_open else np.nan,
            "option_liability": opt_liab,
            "realized_pl": realized_pl,
            "unrealized_pl": unrealized_pl,
            "total_equity": total_equity,
            "benchmark_equity": bench_equity,
            "alpha_vs_benchmark": total_equity - bench_equity,
        })

        redraw(i, total_equity, bench_equity, unrealized_pl, dte_left, delta_now)

        # Expiry handling
        if pos.is_open and i == pos.expiry_idx:
            last_K = float(pos.K)
            decisions.append({
                "date": date,
                "i": i,
                "decision": "EXPIRE",
                "note": f"Call expires today; settle. ITM={S > last_K}",
                "S": S, "iv": iv,
                "K": last_K,
                "dte_left": dte_left,
                "delta_now": delta_now,
                "cash": cash,
                "equity": total_equity,
                "benchmark_equity": bench_equity,
            })

            close_call(i, reason="EXPIRES_TODAY_SETTLE")
            if S > last_K:
                assign_and_rebuy_if_needed(i, strike=last_K)

            if shares > 0 and not pos.is_open:
                open_call(i, reason="RESELL_AFTER_EXPIRY", prompt_params=True)
            continue

        if shares <= 0:
            print(f"{date.date()}  S={S:.2f}  (FLAT)  Cash={money0(cash)}  Eq={money0(total_equity)}")
            cmd = input("[Q]uit -> ").strip().lower()
            decisions.append({
                "date": date, "i": i, "decision": cmd.upper() if cmd else "NONE",
                "note": "Flat; only quit allowed",
                "S": S, "iv": iv, "K": float(pos.K) if pos.is_open else np.nan,
                "dte_left": dte_left, "delta_now": delta_now,
                "cash": cash, "equity": total_equity, "benchmark_equity": bench_equity
            })
            if cmd == "q":
                break
            continue

        if shares > 0 and not pos.is_open:
            print(f"{date.date()}  S={S:.2f}  Cash={money0(cash)}  Eq={money0(total_equity)}")
            cmd = input("[S]ell (prompt)  [SF]ast sell  [K]eep flat  [Q]uit -> ").strip().lower()
            decisions.append({
                "date": date, "i": i, "decision": cmd.upper() if cmd else "NONE",
                "note": "No short call open",
                "S": S, "iv": iv, "K": np.nan,
                "dte_left": 0, "delta_now": np.nan,
                "cash": cash, "equity": total_equity, "benchmark_equity": bench_equity
            })
            if cmd == "q":
                break
            if cmd == "s":
                open_call(i, reason="USER_SELL_NEW_CALL", prompt_params=True)
            elif cmd == "sf":
                open_call(i, reason="USER_SELL_NEW_CALL_FAST", prompt_params=False)
            continue

        status = (
            f"{date.date()}  S={S:.2f}  IV={iv:.2f}  Cash={money0(cash)}  Eq={money0(total_equity)}  "
            f"Real={money0(realized_pl)}  Unrl={money0(unrealized_pl)}  |  "
            f"K={pos.K:.2f}  DTE_left={dte_left}  Δnow={delta_now:.2f}"
        )
        print(status)

        cmd = input("[K]eep  [C]lose  [R]oll FAST  [RR]oll prompt  [Q]uit -> ").strip().lower()

        decisions.append({
            "date": date,
            "i": i,
            "decision": cmd.upper() if cmd else "NONE",
            "note": "Main decision point",
            "S": S, "iv": iv, "K": float(pos.K),
            "dte_left": dte_left, "delta_now": delta_now,
            "cash": cash, "equity": total_equity, "benchmark_equity": bench_equity
        })

        if cmd == "q":
            break
        elif cmd in ("", "k"):
            continue
        elif cmd == "c":
            close_call(i, reason="USER_CLOSE")
            if shares > 0 and not pos.is_open:
                sub = input("Open new call now? [Y] prompt / [F]ast / [N] -> ").strip().lower()
                decisions.append({
                    "date": date,
                    "i": i,
                    "decision": f"POST_CLOSE_{sub.upper() if sub else 'NONE'}",
                    "note": "After closing, chose whether to open new call",
                    "S": S, "iv": iv, "K": np.nan,
                    "dte_left": 0, "delta_now": np.nan,
                    "cash": cash, "equity": total_equity, "benchmark_equity": bench_equity
                })
                if sub in ("", "y"):
                    open_call(i, reason="AFTER_USER_CLOSE", prompt_params=True)
                elif sub == "f":
                    open_call(i, reason="AFTER_USER_CLOSE_FAST", prompt_params=False)

        elif cmd == "r":
            close_call(i, reason="USER_ROLL_CLOSE_FAST")
            if shares > 0 and not pos.is_open:
                open_call(i, reason="USER_ROLL_OPEN_FAST", prompt_params=False)

        elif cmd == "rr":
            close_call(i, reason="USER_ROLL_CLOSE_PROMPT")
            if shares > 0 and not pos.is_open:
                open_call(i, reason="USER_ROLL_OPEN_PROMPT", prompt_params=True)

    plt.ioff()
    return {
        "equity": pd.DataFrame(equity_rows),
        "trades": pd.DataFrame(trades),
        "decisions": pd.DataFrame(decisions),
        "figure": fig
    }


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    regime_name, path_cfg = choose_regime()
    print(f"\nRegime selected: {regime_name}\n")

    print("--- Covered Call Simulator Inputs ---")
    starting_cash = ask_float("Starting cash ($)", 100000.0, lo=0.0)
    default_delta = ask_float("Default delta for new calls", 0.30, lo=0.01, hi=0.99)
    default_dte = ask_int("Default DTE days for new calls", 14, lo=1, hi=365)

    cfg = StrategyConfig(
        shares=100,
        default_delta=default_delta,
        default_dte_days=default_dte,
        r=0.04,
        trading_days=252,
        commission_option=1.00,
        commission_stock=0.00,
        auto_rebuy_after_assignment=True,
        plot_lookback_days=120,
    )

    df = simulate_synthetic_spy(path_cfg)

    out = run_interactive(df, cfg, starting_cash=starting_cash)

    if out["equity"] is not None and not out["equity"].empty:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            out["equity"].to_excel(writer, sheet_name="EquityCurve", index=False)
            out["trades"].to_excel(writer, sheet_name="TradeLog", index=False)
            out["decisions"].to_excel(writer, sheet_name="DecisionLog", index=False)

        print("\nSaved workbook:")
        print(OUTPUT_XLSX)
        print("\nSheets: EquityCurve, TradeLog, DecisionLog")
        print("\nTip: The file is in the repo folder under ./output/")

    print("\nClose the plot window to finish.")
    plt.show()
