import math
from typing import Tuple


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_call_price_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> Tuple[float, float, float, float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, S - K)
        delta = 1.0 if S > K else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    call = disc_q * S * Nd1 - disc_r * K * Nd2
    delta = disc_q * Nd1
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = disc_q * S * pdf_d1 * sqrtT
    theta = (
        -(disc_q * S * pdf_d1 * sigma) / (2 * sqrtT)
        - r * disc_r * K * Nd2
        + q * disc_q * S * Nd1
    )
    return call, delta, gamma, vega, theta


def find_strike_for_target_delta(
    S: float, T: float, r: float, q: float, sigma: float, target_delta: float
) -> float:
    target_delta = max(0.01, min(0.99, target_delta))

    K_low = max(0.1, S * 0.5)
    K_high = S * 1.8

    for _ in range(80):
        K_mid = 0.5 * (K_low + K_high)
        _, d_mid, _, _, _ = bs_call_price_greeks(S, K_mid, T, r, q, sigma)
        if d_mid > target_delta:
            K_low = K_mid
        else:
            K_high = K_mid

    return round(K_high, 2)

def covered_call_strike_for_delta(
    spot: float,
    dte_days: float,
    iv: float,
    target_delta: float,
    r: float = 0.05,
    q: float = 0.0,
) -> float:
    """
    Web-friendly wrapper:
    - dte_days is days to expiration
    - returns strike for a call with the requested target delta
    """
    T = dte_days / 365.0
    return find_strike_for_target_delta(spot, T, r, q, iv, target_delta)
