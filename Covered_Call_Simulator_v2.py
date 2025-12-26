"""
Covered Call Simulator (v2) — single-file script (robust plotting + rolling x-window)

Robust fixes included:
- Persistent Matplotlib figure (no per-step close/recreate) -> avoids TkAgg black rectangles
- Full redraw via canvas.draw() + flush_events()
- Toolbar disabled
- TRUE rolling window: constant number of days shown (lookback), not compressed
- Info box in reserved right margin (never overlaps y-axis labels)
- Strike line is red dashed
- Dot markers for latest underlying price and latest CC total equity
- Handles xmin==xmax on first bar to avoid identical xlim warnings

Dependencies:
  numpy, pandas, matplotlib, scipy, openpyxl
"""

from __future__ import annotations

# -----------------------------
# Force matplotlib to use an external window (not PyCharm SciView)
# IMPORTANT: must come BEFORE importing pyplot.
# -----------------------------
import matplotlib

matplotlib.use("TkAgg")

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

# Disable toolbar (reduces TkAgg redraw artifacts further)
plt.rcParams["toolbar"] = "None"

# ============================================================
# OUTPUT DIRECTORY (general use, cross-platform)
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


def money2(x: float) -> str:
    return f"${x:,.2f}"


def dollars_tick(x, pos=None) -> str:
    return f"${x:,.0f}"


# ============================================================
# Robust input helpers
# ============================================================
def ask_int(prompt: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "":
            return default
        try:
            v = int(s)
            if lo is not None and v < lo:
                print(f"Please enter an integer >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"Please enter an integer <= {hi}.")
                continue
            return v
        except ValueError:
            print("Please enter an integer (or press Enter for default).")


def ask_float(prompt: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    while True:
        s = input(prompt).strip()
        if s == "":
            return default
        try:
            v = float(s)
            if lo is not None and v < lo:
                print(f"Please enter a number >= {lo}.")
                continue
            if hi is not None and v > hi:
                print(f"Please enter a number <= {hi}.")
                continue
            return v
        except ValueError:
            print("Please enter a number (or press Enter for default).")


def ask_choice(prompt: str, choices: Dict[str, str], default_key: str) -> str:
    while True:
        s = input(prompt).strip()
        if s == "":
            return choices[default_key]
        if s in choices:
            return choices[s]
        print(f"Choose one of: {', '.join(choices.keys())} (or press Enter for default).")


# ============================================================
# Black–Scholes (European call, no dividends)
# ============================================================
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)


def strike_for_target_delta(
    S: float, target_delta: float, T: float, r: float, sigma: float, tol: float = 1e-6
) -> float:
    target_delta = max(1e-4, min(0.9999, float(target_delta)))

    K_low = max(0.01, 0.2 * S)
    K_high = 5.0 * S

    def f(K: float) -> float:
        return bs_call_delta(S, K, T, r, sigma) - target_delta

    f_low = f(K_low)
    f_high = f(K_high)

    for _ in range(40):
        if f_low * f_high <= 0:
            break
        K_low *= 0.5
        K_high *= 1.5
        f_low = f(K_low)
        f_high = f(K_high)

    if f_low * f_high > 0:
        return S * (0.95 if target_delta > 0.5 else 1.05 + 1.5 * (0.5 - target_delta))

    for _ in range(200):
        K_mid = 0.5 * (K_low + K_high)
        f_mid = f(K_mid)
        if abs(f_mid) < tol:
            return K_mid
        if f_low * f_mid <= 0:
            K_high = K_mid
            f_high = f_mid
        else:
            K_low = K_mid
            f_low = f_mid

    return 0.5 * (K_low + K_high)


# ============================================================
# Synthetic price generator with regimes
# ============================================================
@dataclass
class PathConfig:
    start_price: float = 450.0
    n_days: int = 252
    seed: int = 1
    base_iv: float = 0.18


def generate_synthetic_spy(regime: str, cfg: PathConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range("2018-01-02", periods=cfg.n_days)

    if regime == "choppy":
        drift, vol, mean_revert, vol_cluster = 0.0000, 0.010, 0.25, 0.25
    elif regime == "uptrending":
        drift, vol, mean_revert, vol_cluster = 0.0005, 0.010, 0.05, 0.20
    elif regime == "downtrending":
        drift, vol, mean_revert, vol_cluster = -0.0005, 0.012, 0.05, 0.25
    elif regime == "volatile":
        drift, vol, mean_revert, vol_cluster = 0.0000, 0.020, 0.10, 0.60
    else:  # mixed
        drift, vol, mean_revert, vol_cluster = 0.0002, 0.013, 0.12, 0.35

    S = np.zeros(cfg.n_days)
    S[0] = cfg.start_price

    iv = np.zeros(cfg.n_days)
    iv[0] = cfg.base_iv
    log_iv = math.log(cfg.base_iv)

    anchor = cfg.start_price

    for t in range(1, cfg.n_days):
        log_iv = (1 - vol_cluster) * math.log(cfg.base_iv) + vol_cluster * log_iv + 0.15 * rng.standard_normal()
        iv[t] = float(np.clip(math.exp(log_iv), 0.08, 0.60))

        mr_term = mean_revert * (math.log(anchor) - math.log(S[t - 1]))
        daily_vol = vol * (iv[t] / cfg.base_iv) ** 0.6
        shock = daily_vol * rng.standard_normal()

        ret = drift + mr_term + shock
        S[t] = S[t - 1] * math.exp(ret)

        anchor = 0.995 * anchor + 0.005 * S[t]

    return pd.DataFrame({"date": dates, "S": S, "iv": iv})


# ============================================================
# Strategy config/state
# ============================================================
@dataclass
class StrategyConfig:
    shares: int = 100
    default_delta: float = 0.30
    default_dte_days: int = 14
    r: float = 0.035
    commission_option: float = 1.00
    commission_stock: float = 0.00


@dataclass
class ShortCall:
    K: float
    opened_idx: int
    expiry_idx: int
    premium: float
    target_delta: float


@dataclass
class State:
    cash: float
    shares: int
    realized_pl: float
    short_call: Optional[ShortCall]
    bh_shares: float
    bh_cash: float


# ============================================================
# Trade and decision logs
# ============================================================
def log_trade(trades: List[dict], date, action: str, details: str, cash_delta: float, cash_after: float):
    trades.append(
        {"Date": pd.to_datetime(date), "Action": action, "Details": details, "CashDelta": cash_delta, "CashAfter": cash_after}
    )


def log_decision(decisions: List[dict], date, decision: str, note: str):
    decisions.append({"Date": pd.to_datetime(date), "Decision": decision, "Note": note})


# ============================================================
# Portfolio accounting helpers
# ============================================================
def short_call_liability(S: float, call: ShortCall, idx: int, df: pd.DataFrame, cfg: StrategyConfig) -> Tuple[float, float]:
    iv = float(df.loc[idx, "iv"])
    T = max(call.expiry_idx - idx, 0) / 252.0
    price = bs_call_price(S, call.K, T, cfg.r, iv) * cfg.shares
    delta = bs_call_delta(S, call.K, T, cfg.r, iv)
    return price, delta


def total_equity(S: float, state: State, call_liab: float) -> float:
    return state.cash + state.shares * S - call_liab


def buy_hold_equity(S: float, state: State) -> float:
    return state.bh_cash + state.bh_shares * S


# ============================================================
# Opening/closing/rolling calls
# ============================================================
def open_new_call(
    df: pd.DataFrame,
    idx: int,
    state: State,
    cfg: StrategyConfig,
    target_delta: float,
    dte_days: int,
    trades: List[dict],
    note: str = "",
) -> None:
    S = float(df.loc[idx, "S"])
    iv = float(df.loc[idx, "iv"])
    expiry_idx = min(idx + dte_days, len(df) - 1)
    T = max(expiry_idx - idx, 1) / 252.0

    K = strike_for_target_delta(S, target_delta, T, cfg.r, iv)
    premium_per_share = bs_call_price(S, K, T, cfg.r, iv)
    premium = premium_per_share * cfg.shares

    state.cash += premium
    state.cash -= cfg.commission_option

    state.short_call = ShortCall(K=K, opened_idx=idx, expiry_idx=expiry_idx, premium=premium, target_delta=target_delta)

    log_trade(
        trades,
        df.loc[idx, "date"],
        "SELL_CALL",
        f"K={K:,.2f}, DTE={expiry_idx - idx}, targetΔ={target_delta:.2f}, prem={money2(premium)} {note}".strip(),
        cash_delta=premium - cfg.commission_option,
        cash_after=state.cash,
    )


def close_call(df: pd.DataFrame, idx: int, state: State, cfg: StrategyConfig, trades: List[dict], reason: str = "") -> None:
    if state.short_call is None:
        return
    S = float(df.loc[idx, "S"])
    call_price, _ = short_call_liability(S, state.short_call, idx, df, cfg)

    state.cash -= call_price
    state.cash -= cfg.commission_option

    log_trade(
        trades,
        df.loc[idx, "date"],
        "BUY_TO_CLOSE",
        f"K={state.short_call.K:,.2f}, cost={money2(call_price)} {reason}".strip(),
        cash_delta=-(call_price + cfg.commission_option),
        cash_after=state.cash,
    )

    state.realized_pl += (state.short_call.premium - call_price) - (2 * cfg.commission_option)
    state.short_call = None


def maybe_handle_expiry_assignment(df: pd.DataFrame, idx: int, state: State, cfg: StrategyConfig, trades: List[dict]) -> None:
    call = state.short_call
    if call is None or idx != call.expiry_idx:
        return

    S = float(df.loc[idx, "S"])
    if S > call.K:
        proceeds = call.K * cfg.shares
        state.cash += proceeds
        state.cash -= cfg.commission_stock

        log_trade(
            trades,
            df.loc[idx, "date"],
            "ASSIGNED",
            f"Shares called away at K={call.K:,.2f}, proceeds={money2(proceeds)}",
            cash_delta=proceeds - cfg.commission_stock,
            cash_after=state.cash,
        )

        intrinsic = max(S - call.K, 0.0) * cfg.shares
        state.realized_pl += (call.premium - intrinsic) - cfg.commission_option

        state.shares = 0
        state.short_call = None
    else:
        log_trade(
            trades,
            df.loc[idx, "date"],
            "EXPIRE",
            f"Call expired worthless (K={call.K:,.2f})",
            cash_delta=0.0,
            cash_after=state.cash,
        )
        state.realized_pl += call.premium - cfg.commission_option
        state.short_call = None


def ensure_shares_or_warn(df: pd.DataFrame, idx: int, state: State, cfg: StrategyConfig) -> bool:
    if state.shares >= cfg.shares:
        return True

    S = float(df.loc[idx, "S"])
    needed = cfg.shares - state.shares
    cost = needed * S

    if state.cash < cost:
        print(f"\n*** Not enough cash to buy {needed} shares at {money2(S)} (need {money2(cost)}, have {money2(state.cash)}).")
        print("*** The simulator will continue, but cannot open a new covered call until you have shares.\n")
        return False

    state.cash -= cost
    state.cash -= cfg.commission_stock
    state.shares += needed
    return True


def buy_shares_if_called(df: pd.DataFrame, idx: int, state: State, cfg: StrategyConfig, trades: List[dict]) -> None:
    if state.shares >= cfg.shares:
        return

    S = float(df.loc[idx, "S"])
    cost = cfg.shares * S

    if state.cash < cost:
        print(f"\n*** Assigned and you do NOT have enough cash to rebuy 100 shares at {money2(S)}.")
        print(f"*** Need {money2(cost)}, have {money2(state.cash)}. You will remain in cash.\n")
        log_trade(
            trades,
            df.loc[idx, "date"],
            "REBUY_FAILED",
            f"Insufficient cash to rebuy 100 shares at {money2(S)}",
            cash_delta=0.0,
            cash_after=state.cash,
        )
        return

    state.cash -= cost
    state.cash -= cfg.commission_stock
    state.shares = cfg.shares

    log_trade(
        trades,
        df.loc[idx, "date"],
        "BUY_SHARES",
        f"Rebought 100 shares at {money2(S)} (after assignment)",
        cash_delta=-(cost + cfg.commission_stock),
        cash_after=state.cash,
    )


# ============================================================
# Persistent plotter (robust + true rolling window + markers)
# ============================================================
class PersistentPlotter:
    def __init__(self, lookback: int = 120):
        plt.ion()
        self.lookback = int(lookback)

        self.fig = plt.figure(figsize=(12.5, 7.5))
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.15)

        self.ax_price = self.fig.add_subplot(self.gs[0, 0])
        self.ax_cash = self.ax_price.twinx()
        self.ax_eq = self.fig.add_subplot(self.gs[1, 0], sharex=self.ax_price)

        # Leave space on the right for the info box (without crushing the plot)
        self.fig.subplots_adjust(right=0.65, top=0.92, bottom=0.10)

        # Lines (created once)
        (self.line_price,) = self.ax_price.plot([], [], linewidth=2, label="Price")
        (self.line_cash,) = self.ax_cash.plot([], [], linestyle="--", linewidth=2, label="Cash")

        # Strike line (red dashed)
        self.strike_line = self.ax_price.axhline(np.nan, color="red", linestyle="--", linewidth=2, label="Strike")

        (self.line_eq,) = self.ax_eq.plot([], [], linewidth=2, label="Covered Call Equity")
        (self.line_bh,) = self.ax_eq.plot([], [], linewidth=2, label="Buy & Hold Equity")

        # Dot markers for the latest point (created once)
        (self.dot_price,) = self.ax_price.plot([], [], marker="o", linestyle="None", markersize=7, label="_nolegend_")
        (self.dot_eq,) = self.ax_eq.plot([], [], marker="o", linestyle="None", markersize=7, label="_nolegend_")

        self.ax_cash.yaxis.set_major_formatter(FuncFormatter(dollars_tick))
        self.ax_eq.yaxis.set_major_formatter(FuncFormatter(dollars_tick))

        self.ax_price.set_ylabel("Price")
        self.ax_cash.set_ylabel("Cash")
        self.ax_eq.set_ylabel("Equity ($)")
        self.ax_eq.set_xlabel("Date")

        # ---- X ticks: readable, non-compressing ----
        # Show x labels ONLY on bottom subplot to avoid crowding
        self.ax_price.tick_params(axis="x", which="both", labelbottom=False)

        self.locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        self.formatter = mdates.ConciseDateFormatter(self.locator)
        self.ax_eq.xaxis.set_major_locator(self.locator)
        self.ax_eq.xaxis.set_major_formatter(self.formatter)

        # Info box anchored just outside the top axes (right side)
        BOX_X = 1.15  # increase (e.g., 1.10) to push further right
        BOX_Y = 0.86
        self.info_text = self.ax_price.text(
            BOX_X,
            BOX_Y,
            "",
            transform=self.ax_price.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.95),
            clip_on=False,
        )

        self._refresh_legends()

    def _refresh_legends(self):
        lines1, labels1 = self.ax_price.get_legend_handles_labels()
        lines2, labels2 = self.ax_cash.get_legend_handles_labels()
        self.ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        self.ax_eq.legend(loc="upper left")

    def update(
        self,
        dates: pd.Series,
        prices: np.ndarray,
        cash: np.ndarray,
        eq: np.ndarray,
        bh: np.ndarray,
        strike_value: float,
        title: str,
        info_lines: List[str],
        xmin: pd.Timestamp,
        xmax: pd.Timestamp,
    ):
        # Update data
        self.line_price.set_data(dates, prices)
        self.line_cash.set_data(dates, cash)
        self.line_eq.set_data(dates, eq)
        self.line_bh.set_data(dates, bh)

        # Strike line
        if np.isnan(strike_value):
            self.strike_line.set_ydata([np.nan, np.nan])
        else:
            self.strike_line.set_ydata([strike_value, strike_value])

        # Latest markers
        self.dot_price.set_data([dates.iloc[-1]], [prices[-1]])
        self.dot_eq.set_data([dates.iloc[-1]], [eq[-1]])

        # Rolling x window limits
        self.ax_price.set_xlim(xmin, xmax)
        self.ax_cash.set_xlim(xmin, xmax)
        self.ax_eq.set_xlim(xmin, xmax)

        # Re-apply locator/formatter (safe as xlim changes)
        self.ax_eq.xaxis.set_major_locator(self.locator)
        self.ax_eq.xaxis.set_major_formatter(self.formatter)

        # Autoscale y only
        self.ax_price.relim()
        self.ax_price.autoscale_view(scalex=False, scaley=True)
        self.ax_cash.relim()
        self.ax_cash.autoscale_view(scalex=False, scaley=True)
        self.ax_eq.relim()
        self.ax_eq.autoscale_view(scalex=False, scaley=True)

        # Title + info
        self.ax_price.set_title(title, fontsize=14)
        self.info_text.set_text("\n".join(info_lines))

        self._refresh_legends()

        # Robust full redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)  # close the figure immediately (no blocking show())


# ============================================================
# Main interactive loop
# ============================================================
def run_interactive(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    regime_name: str,
    starting_cash: float,
    lookback: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades: List[dict] = []
    decisions: List[dict] = []

    state = State(
        cash=starting_cash,
        shares=0,
        realized_pl=0.0,
        short_call=None,
        bh_shares=(starting_cash / float(df.loc[0, "S"])),
        bh_cash=0.0,
    )

    # Buy initial 100 shares (once)
    if ensure_shares_or_warn(df, 0, state, cfg):
        S0 = float(df.loc[0, "S"])
        cost = cfg.shares * S0
        log_trade(
            trades,
            df.loc[0, "date"],
            "BUY_SHARES",
            f"Bought {cfg.shares} shares at {money2(S0)}",
            cash_delta=-(cost + cfg.commission_stock),
            cash_after=state.cash,
        )

    if state.shares >= cfg.shares:
        open_new_call(df, 0, state, cfg, cfg.default_delta, cfg.default_dte_days, trades, note="(initial)")

    plotter = PersistentPlotter(lookback=lookback)

    eq_rows: List[dict] = []
    for idx in range(len(df)):
        date = df.loc[idx, "date"]
        S = float(df.loc[idx, "S"])

        maybe_handle_expiry_assignment(df, idx, state, cfg, trades)
        if state.shares == 0:
            buy_shares_if_called(df, idx, state, cfg, trades)

        liab = 0.0
        delta_now = np.nan
        dte_left = 0
        dte_opened = np.nan
        delta_target = np.nan

        if state.short_call:
            liab, delta_now = short_call_liability(S, state.short_call, idx, df, cfg)
            dte_left = max(state.short_call.expiry_idx - idx, 0)
            dte_opened = state.short_call.expiry_idx - state.short_call.opened_idx
            delta_target = state.short_call.target_delta

        eq_val = total_equity(S, state, liab)
        bh_val = buy_hold_equity(S, state)
        unrl = (state.short_call.premium - liab) if state.short_call else 0.0

        eq_rows.append(
            {
                "Date": pd.to_datetime(date),
                "S": S,
                "IV": float(df.loc[idx, "iv"]),
                "Cash": state.cash,
                "Shares": state.shares,
                "CallStrike": (state.short_call.K if state.short_call else np.nan),
                "CallDTE_left": dte_left,
                "CallDelta_now": (float(delta_now) if state.short_call else np.nan),
                "CallLiability": liab,
                "Eq": eq_val,
                "BH": bh_val,
                "Real": state.realized_pl,
                "Unrl": unrl,
            }
        )

        equity_df = pd.DataFrame(eq_rows)

        # --- TRUE rolling window: only last `lookback` points are plotted
        start = max(0, idx - lookback + 1)
        end = idx + 1

        d = pd.to_datetime(df["date"].iloc[start:end])
        S_series = df["S"].iloc[start:end].astype(float).to_numpy()

        cash_series = equity_df["Cash"].iloc[start:end].astype(float).to_numpy()
        eq_series = equity_df["Eq"].iloc[start:end].astype(float).to_numpy()
        bh_series = equity_df["BH"].iloc[start:end].astype(float).to_numpy()

        xmin = d.iloc[0]
        xmax = d.iloc[-1]
        if xmin == xmax:
            xmin = xmin - pd.Timedelta(days=1)
            xmax = xmax + pd.Timedelta(days=1)

        title = (
            f"Eq={money0(eq_val)} | BH={money0(bh_val)} | "
            f"Real={money0(state.realized_pl)} | Unrl={money0(unrl)} | "
            f"Cash={money0(state.cash)} | DTE_left={dte_left}"
        )

        info_lines = [f"Regime: {regime_name}", f"S: {money2(S)}"]
        if state.short_call:
            info_lines += [
                f"DTE_left: {dte_left}",
                f"Δ now: {delta_now:.2f}",
                f"Strike: {money2(state.short_call.K)}",
                f"Δ target: {delta_target:.2f}",
                f"DTE(opened): {int(dte_opened)}",
            ]
            strike_val = float(state.short_call.K)
        else:
            info_lines += ["(No short call open)"]
            strike_val = np.nan

        plotter.update(
            dates=d,
            prices=S_series,
            cash=cash_series,
            eq=eq_series,
            bh=bh_series,
            strike_value=strike_val,
            title=title,
            info_lines=info_lines,
            xmin=xmin,
            xmax=xmax,
        )

        # Console status + decision prompt
        if state.short_call:
            print(
                f"{equity_df.loc[idx, 'Date'].date()}  S={money2(S)}  IV={equity_df.loc[idx, 'IV']:.2f}  "
                f"Cash={money0(state.cash)}  TotalEq={money0(eq_val)} |  "
                f"ShortCall K={money2(state.short_call.K)}  DaysToExp={dte_left}  "
                f"Δ={float(delta_now):.2f}  Liab={money0(liab)}"
            )
            prompt = "[K]eep  [C]lose  [R]oll(fast)  [RR]oll+Change  [Q]uit -> "
        else:
            print(
                f"{equity_df.loc[idx, 'Date'].date()}  S={money2(S)}  IV={equity_df.loc[idx, 'IV']:.2f}  "
                f"Cash={money0(state.cash)}  TotalEq={money0(eq_val)} |  (No short call)"
            )
            prompt = "[K]eep  [RR]oll+Change (open new)  [Q]uit -> "

        cmd = input(prompt).strip().upper()
        if cmd == "":
            cmd = "K"

        if cmd in ["KEEP"]:
            cmd = "K"
        if cmd in ["CLOSE"]:
            cmd = "C"
        if cmd in ["ROLL", "R"]:
            cmd = "R"
        if cmd in ["RR", "ROLLCHANGE", "ROLL+CHANGE"]:
            cmd = "RR"
        if cmd in ["QUIT", "Q"]:
            cmd = "Q"

        if cmd == "Q":
            log_decision(decisions, date, "Q", "Quit")
            break

        if cmd == "C":
            if state.short_call:
                close_call(df, idx, state, cfg, trades, reason="(user close)")
            log_decision(decisions, date, "C", "Close call")

        elif cmd == "R":
            if state.short_call:
                close_call(df, idx, state, cfg, trades, reason="(fast roll close)")
                if ensure_shares_or_warn(df, idx, state, cfg):
                    open_new_call(df, idx, state, cfg, cfg.default_delta, cfg.default_dte_days, trades, note="(fast roll open)")
                else:
                    print("*** Cannot open a covered call without shares.")
            else:
                print("*** No call to roll.")
            log_decision(decisions, date, "R", "Fast roll (defaults)")

        elif cmd == "RR":
            new_delta = ask_float(
                f"Default call delta target (0.05..0.60, default {cfg.default_delta:.2f}): ",
                default=cfg.default_delta, lo=0.05, hi=0.60
            )
            new_dte = ask_int(
                f"When opening a new covered call, how many trading days until it expires? "
                f"(Enter a number between 5 and 60; default is {cfg.default_dte_days}): ",
                default=cfg.default_dte_days, lo=5, hi=60
            )
            cfg.default_delta = float(new_delta)
            cfg.default_dte_days = int(new_dte)

            if state.short_call:
                close_call(df, idx, state, cfg, trades, reason="(roll+change close)")
            if ensure_shares_or_warn(df, idx, state, cfg):
                open_new_call(df, idx, state, cfg, cfg.default_delta, cfg.default_dte_days, trades, note="(roll+change open)")
            else:
                print("*** Cannot open a covered call without shares.")

            log_decision(decisions, date, "RR", f"Roll+Change (delta={cfg.default_delta:.2f}, dte={cfg.default_dte_days})")

        else:
            log_decision(decisions, date, "K", "Keep")

    plotter.close()

    equity_df = pd.DataFrame(eq_rows)
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame(decisions)
    return equity_df, trades_df, decisions_df


# ============================================================
# Regime chooser
# ============================================================
def choose_regime() -> Tuple[str, PathConfig]:
    print("\nCovered Call Simulator (v2)\n")
    print("Choose a regime:")
    print("  1) Choppy  (choppy)")
    print("  2) Uptrending  (uptrending)")
    print("  3) Downtrending  (downtrending)")
    print("  4) Volatile  (volatile)")
    print("  5) Mixed  (mixed)")

    regime_map = {"1": "choppy", "2": "uptrending", "3": "downtrending", "4": "volatile", "5": "mixed"}
    regime = ask_choice("Regime number (default 1): ", regime_map, default_key="1")

    start_price = ask_float("Start price for synthetic 'SPY' (default 450): ", default=450.0, lo=10.0, hi=5000.0)
    n_days = ask_int("Number of trading days to simulate (default 252): ", default=252, lo=20, hi=5000)
    seed = ask_int("Random seed (integer, default 1): ", default=1, lo=0, hi=10_000_000)

    cfg = PathConfig(start_price=float(start_price), n_days=int(n_days), seed=int(seed), base_iv=0.18)
    return regime, cfg


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    regime, path_cfg = choose_regime()
    regime_name = regime.capitalize()

    r = ask_float("Risk-free rate (annual, default 3.5%): ", default=0.035, lo=0.0, hi=0.20)

    starting_cash = ask_float(
        "Starting cash (must be enough to buy 100 shares to sell covered calls; default 100000): ",
        default=100000.0, lo=0.0, hi=1e9
    )

    default_delta = ask_float(
        "Default call delta target (0.05..0.60, default 0.30): ",
        default=0.30, lo=0.05, hi=0.60
    )

    default_dte = ask_int(
        "When opening a new covered call, how many trading days until it expires? (Enter 5 to 60; default 14): ",
        default=14, lo=5, hi=60
    )

    cfg = StrategyConfig(
        shares=100,
        default_delta=float(default_delta),
        default_dte_days=int(default_dte),
        r=float(r),
        commission_option=1.00,
        commission_stock=0.00,
    )

    df = generate_synthetic_spy(regime, path_cfg)

    print("\nGenerating synthetic price path and opening the initial call...\n")

    equity_df, trades_df, decisions_df = run_interactive(df, cfg, regime_name, starting_cash=starting_cash, lookback=120)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        equity_df.to_excel(writer, sheet_name="EquityCurve", index=False)
        trades_df.to_excel(writer, sheet_name="TradeLog", index=False)
        decisions_df.to_excel(writer, sheet_name="DecisionLog", index=False)

    print("\nSaved workbook:")
    print(OUTPUT_XLSX)
    print("\nSheets: EquityCurve, TradeLog, DecisionLog")
    print("\nDone.")

