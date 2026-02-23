# app.py
# Wave Formula Fitter + Hurst/Mean Reversion Explorer
#
# Update: supports selecting from MULTIPLE CSV files in the same folder (commodities)
# - Choose local CSV from folder OR upload
# - Then pick column, timeframe, harmonics, forward period
#
import os
import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Safety / defaults
# -----------------------------
st.set_page_config(page_title="Commodity Waves: Wave Formula + Hurst", layout="wide")

DEFAULT_CSV_NAME = "SPX.csv"
MAX_POINTS_FOR_FIT = 25_000  # guardrail for speed
EPS = 1e-12

# When scanning folder for CSVs, ignore these common generated outputs
IGNORE_LOCAL_CSV_NAMES = {
    "wave_fit_forecast.csv",
    "results.csv",
    "output.csv",
}


# -----------------------------
# Helpers
# -----------------------------
def human_int(n: int) -> str:
    return f"{int(n):,}"


def list_local_csvs(folder: str = ".") -> List[str]:
    """
    List CSV files next to app.py (same repo folder).
    """
    out = []
    try:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".csv") and fn not in IGNORE_LOCAL_CSV_NAMES:
                out.append(fn)
    except Exception:
        return []
    out = sorted(out, key=lambda s: s.lower())

    # Prefer SPX.csv at the top if present
    if DEFAULT_CSV_NAME in out:
        out.remove(DEFAULT_CSV_NAME)
        out.insert(0, DEFAULT_CSV_NAME)
    return out


def parse_csv(file_bytes: bytes, filename: str = "data.csv") -> pd.DataFrame:
    """
    Reads CSV bytes, finds a date-like column, parses it, sets index to datetime.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]

    # find date column (more robust)
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "datetime", "timestamp", "time"):
            date_col = c
            break
    if date_col is None:
        # fallback: any column containing "date" is acceptable
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
    if date_col is None:
        raise ValueError("Could not find a Date column (expected Date/Datetime/Timestamp/Time).")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    # drop fully empty cols
    df = df.dropna(axis=1, how="all")
    return df


def read_local_csv_bytes(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()


def robust_column_choices(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def downsample_if_needed(y: np.ndarray, t: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n <= max_points:
        return y, t
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return y[idx], t[idx]


def make_time_index(n: int, dt: float = 1.0) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


# -----------------------------
# Fourier series fit with trend
# y(t) ~ a0 + a1*t + sum_k [ b_k cos(2πk t / T) + c_k sin(2πk t / T) ]
# -----------------------------
@dataclass
class WaveFitResult:
    T: float
    harmonics: int
    coef: np.ndarray  # [a0, a1, b1..bH, c1..cH]
    y_mean: float
    y_std: float
    rmse: float
    r2: float


def build_design_matrix(t: np.ndarray, T: float, H: int) -> np.ndarray:
    n = len(t)
    X = np.empty((n, 2 + 2 * H), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = t
    w = 2.0 * np.pi / T
    for k in range(1, H + 1):
        X[:, 2 + (k - 1)] = np.cos(w * k * t)
        X[:, 2 + H + (k - 1)] = np.sin(w * k * t)
    return X


def fit_wave_formula(y: np.ndarray, harmonics: int) -> WaveFitResult:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 10:
        raise ValueError("Not enough data points in selected timeframe.")

    y_mean = float(y.mean())
    y_std = float(y.std() + EPS)
    y0 = (y - y_mean) / y_std

    t = make_time_index(n, dt=1.0)

    # Base period = window length in steps (captures full-window cycles)
    T = float(max(2.0, n - 1))
    H = int(max(1, harmonics))

    X = build_design_matrix(t, T=T, H=H)
    coef, *_ = np.linalg.lstsq(X, y0, rcond=None)
    yhat0 = X @ coef
    resid = y0 - yhat0

    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y0 - y0.mean()) ** 2) + EPS)
    r2 = float(1.0 - ss_res / ss_tot)

    return WaveFitResult(T=T, harmonics=H, coef=coef.astype(np.float64),
                         y_mean=y_mean, y_std=y_std, rmse=rmse, r2=r2)


def eval_wave_formula(fit: WaveFitResult, n_points: int, start_t: float = 0.0) -> np.ndarray:
    t = start_t + make_time_index(n_points, dt=1.0)
    X = build_design_matrix(t, T=fit.T, H=fit.harmonics)
    y0 = X @ fit.coef
    y = y0 * fit.y_std + fit.y_mean
    return y.astype(np.float64)


def wave_formula_text(fit: WaveFitResult) -> str:
    a0 = fit.coef[0]
    a1 = fit.coef[1]
    H = fit.harmonics
    bs = fit.coef[2:2 + H]
    cs = fit.coef[2 + H:2 + 2 * H]

    lines = []
    lines.append("y(t) = mean + std * [ a0 + a1*t + Σ_{k=1..H}( b_k cos(2πk t / T) + c_k sin(2πk t / T) ) ]")
    lines.append("")
    lines.append(f"mean = {fit.y_mean:.6g}")
    lines.append(f"std  = {fit.y_std:.6g}")
    lines.append(f"T    = {fit.T:.6g} (window length in steps)")
    lines.append(f"H    = {fit.harmonics}")
    lines.append("")
    lines.append(f"a0 = {a0:.6g}")
    lines.append(f"a1 = {a1:.6g}")
    show = min(H, 20)
    for k in range(1, show + 1):
        lines.append(f"b{k} = {bs[k-1]:.6g}    c{k} = {cs[k-1]:.6g}")
    if H > show:
        lines.append(f"... ({H - show} more harmonics not shown)")
    return "\n".join(lines)


# -----------------------------
# Hurst exponent estimators
# -----------------------------
def hurst_rs(series: np.ndarray, min_chunk: int = 16, max_chunk: int = 512) -> Tuple[float, pd.DataFrame]:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < max(64, min_chunk * 4):
        return float("nan"), pd.DataFrame()

    sizes = []
    s = min_chunk
    while s <= min(max_chunk, N // 2):
        sizes.append(s)
        s *= 2

    rows = []
    for n in sizes:
        m = N // n
        if m < 2:
            continue
        rs_vals = []
        for i in range(m):
            seg = x[i*n:(i+1)*n]
            seg = seg - seg.mean()
            z = np.cumsum(seg)
            R = z.max() - z.min()
            S = seg.std() + EPS
            rs_vals.append(R / S)
        rs_mean = float(np.mean(rs_vals))
        rows.append({"chunk": n, "RS_mean": rs_mean})

    df = pd.DataFrame(rows)
    if df.empty or df["RS_mean"].le(0).all():
        return float("nan"), df

    lx = np.log(df["chunk"].values.astype(np.float64))
    ly = np.log(df["RS_mean"].values.astype(np.float64) + EPS)
    A = np.vstack([np.ones_like(lx), lx]).T
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    H = float(beta[1])
    return H, df


def mean_reversion_stats(price: np.ndarray) -> Dict[str, Any]:
    p = np.asarray(price, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 50:
        return {"ok": False, "reason": "Not enough points for mean-reversion stats (need ~50+)."}

    logp = np.log(np.maximum(p, EPS))
    r = np.diff(logp)

    r1 = r[1:]
    r0 = r[:-1]
    X = np.vstack([np.ones_like(r0), r0]).T
    beta, *_ = np.linalg.lstsq(X, r1, rcond=None)
    c_ret = float(beta[0])
    phi_ret = float(beta[1])

    x = logp - logp.mean()
    x1 = x[1:]
    x0 = x[:-1]
    Xp = np.vstack([np.ones_like(x0), x0]).T
    bp, *_ = np.linalg.lstsq(Xp, x1, rcond=None)
    a_p = float(bp[0])
    b_p = float(bp[1])

    half_life = float("nan")
    if 0.0 < b_p < 1.0:
        half_life = float(-math.log(2.0) / math.log(b_p))

    return {
        "ok": True,
        "phi_returns": phi_ret,
        "c_returns": c_ret,
        "b_logprice": b_p,
        "a_logprice": a_p,
        "half_life_days": half_life,
        "interpretation": (
            "phi_returns < 0 suggests short-term mean reversion (oscillatory returns); "
            "phi_returns > 0 suggests short-term momentum in returns. "
            "b_logprice closer to 1 implies slow mean reversion in log-price deviations."
        ),
    }


# -----------------------------
# UI
# -----------------------------
st.title("CSV → Wave Formula (Fourier) + Hurst Exponent + Mean Reversion")

st.markdown(
    "This app fits a **wave formula** to a selected column over a selected timeframe, then "
    "projects forward using the **same fitted formula**. It also computes the **Hurst exponent** "
    "and **mean-reversion diagnostics**.\n\n"
    "**Wave formula (Fourier series + trend):**\n"
    r"$y(t) = \mu + \sigma\left[a_0 + a_1 t + \sum_{k=1}^{H}\left(b_k \cos\left(\frac{2\pi k t}{T}\right)+c_k \sin\left(\frac{2\pi k t}{T}\right)\right)\right]$"
)

with st.sidebar:
    st.header("1) Data source")

    local_csvs = list_local_csvs(".")
    source = st.radio("CSV source", ["Pick CSV from app folder", "Upload CSV"], index=0)

    csv_bytes = None
    filename = "data.csv"

    if source == "Pick CSV from app folder":
        if not local_csvs:
            st.error("No .csv files found next to app.py. Add commodity CSVs or choose Upload.")
        else:
            chosen = st.selectbox("Choose local CSV", local_csvs, index=0)
            try:
                csv_bytes = read_local_csv_bytes(chosen)
                filename = chosen
            except Exception as e:
                st.error(f"Could not read {chosen}: {e}")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            csv_bytes = up.getvalue()
            filename = getattr(up, "name", "uploaded.csv")

    st.divider()
    st.header("2) Fit settings")
    harmonics = st.slider("Harmonics (H)", min_value=1, max_value=200, value=30, step=1)
    max_points = st.select_slider("Max points used for fit (speed cap)", options=[2_000, 5_000, 10_000, 25_000, 50_000], value=25_000)
    forward_days = st.select_slider("Look-forward period (days)", options=[5, 10, 21, 42, 63, 126, 252, 504], value=126)

    st.divider()
    st.header("3) Hurst settings")
    hurst_min = st.select_slider("Min chunk size", options=[8, 16, 32, 64], value=16)
    hurst_max = st.select_slider("Max chunk size", options=[128, 256, 512, 1024, 2048], value=512)

if csv_bytes is None:
    st.stop()

# Load data
try:
    df = parse_csv(csv_bytes, filename=filename)
except Exception as e:
    st.error(f"Failed to read CSV ({filename}): {e}")
    st.stop()

cols = robust_column_choices(df)
if not cols:
    st.error("No numeric columns found.")
    st.stop()

# Timeframe selection
min_date = df.index.min().date()
max_date = df.index.max().date()

st.caption(f"Loaded file: **{filename}**  |  Rows: **{human_int(len(df))}**  |  Date range: **{min_date} → {max_date}**")

cA, cB, cC = st.columns([1, 1, 1])
with cA:
    default_col = 0
    for preferred in ("Close", "Adj Close", "Settle", "Settlement", "Last", "Price"):
        if preferred in cols:
            default_col = cols.index(preferred)
            break
    col = st.selectbox("Select column", cols, index=default_col)
with cB:
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with cC:
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

if start_date >= end_date:
    st.warning("Start date must be before end date.")
    st.stop()

sub = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
if sub.empty:
    st.warning("No rows in selected timeframe.")
    st.stop()

y = sub[col].astype(float).values
dates = sub.index

# Show overview
st.subheader("Selected series")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows in timeframe", human_int(len(sub)))
m2.metric("Start", str(dates.min().date()))
m3.metric("End", str(dates.max().date()))
m4.metric("Column", col)

fig = plt.figure(figsize=(10, 3))
plt.plot(dates, y)
plt.title(f"{filename} — {col} over selected timeframe")
plt.xlabel("Date")
plt.ylabel(col)
plt.tight_layout()
st.pyplot(fig)

# Fit button
st.divider()
fit_btn = st.button("Fit wave formula", type="primary")

if "fit_result" not in st.session_state:
    st.session_state["fit_result"] = None
    st.session_state["fit_meta"] = None
    st.session_state["fit_file"] = None

if fit_btn:
    try:
        y_fit, t_fit = downsample_if_needed(y, make_time_index(len(y)), int(max_points))
        fit = fit_wave_formula(y_fit, harmonics=int(harmonics))
        st.session_state["fit_result"] = fit
        st.session_state["fit_file"] = filename
        st.session_state["fit_meta"] = {
            "n_points_used": int(len(y_fit)),
            "n_points_total": int(len(y)),
            "harmonics": int(harmonics),
            "forward_days": int(forward_days),
            "column": col,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
        st.success("Wave formula fitted.")
    except Exception as e:
        st.error(f"Fit failed: {e}")

fit: Optional[WaveFitResult] = st.session_state.get("fit_result", None)
meta = st.session_state.get("fit_meta", None)

if fit is None:
    st.info("Click **Fit wave formula** to compute the wave formula and forecast.")
    st.stop()

# Evaluate fitted wave on window
n_window = len(y)
yhat = eval_wave_formula(fit, n_points=n_window, start_t=0.0)

# Forecast forward
n_fwd = int(forward_days)
y_fwd = eval_wave_formula(fit, n_points=n_fwd, start_t=float(n_window))

# Infer frequency
if len(dates) >= 2:
    median_dt = np.median(np.diff(dates.values).astype("timedelta64[D]").astype(int))
else:
    median_dt = 1
freq = "B" if median_dt <= 2 else "D"
fwd_dates = pd.date_range(dates.max(), periods=n_fwd + 1, freq=freq)[1:]

# Plot fit + forecast
st.subheader("Wave fit + forward projection (same formula)")
fig2 = plt.figure(figsize=(12, 4))
plt.plot(dates, y, label="Actual")
plt.plot(dates, yhat, label="Wave fit")
plt.plot(fwd_dates, y_fwd, label="Forward (same wave)")
plt.axvline(dates.max(), linestyle="--", linewidth=1)
plt.title(f"{filename} — Wave fit (H={fit.harmonics}) + forward projection ({n_fwd} steps)")
plt.xlabel("Date")
plt.ylabel(col)
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# Fit quality metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE (normalized)", f"{fit.rmse:.4f}")
c2.metric("R² (normalized)", f"{fit.r2:.4f}")
c3.metric("Harmonics", str(fit.harmonics))
c4.metric("Points used for fit", f"{meta['n_points_used']} / {meta['n_points_total']}")

with st.expander("Show wave formula coefficients"):
    st.text(wave_formula_text(fit))

# Hurst + mean reversion
st.divider()
st.subheader("Hurst exponent + mean reversion diagnostics")

hurst_mode = st.radio("Hurst series", ["Log price", "Log returns"], horizontal=True, index=0)
p = np.asarray(y, dtype=np.float64)
logp = np.log(np.maximum(p, EPS))
series = logp if hurst_mode == "Log price" else np.diff(logp)

H, df_h = hurst_rs(series, min_chunk=int(hurst_min), max_chunk=int(hurst_max))
mr = mean_reversion_stats(p)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Hurst (R/S)", "NaN" if not np.isfinite(H) else f"{H:.3f}")
h2.metric("Interpretation", "—" if not np.isfinite(H) else ("Mean reverting" if H < 0.5 else ("Random-walk-ish" if abs(H-0.5) < 0.05 else "Trending")))
if mr.get("ok"):
    h3.metric("AR(1) phi on returns", f"{mr['phi_returns']:.3f}")
    h4.metric("Half-life (log-price dev)", "NaN" if not np.isfinite(mr["half_life_days"]) else f"{mr['half_life_days']:.1f} d")
else:
    h3.metric("AR(1) phi on returns", "—")
    h4.metric("Half-life", "—")

if mr.get("ok"):
    st.caption(mr["interpretation"])
else:
    st.warning(mr.get("reason", "Mean reversion stats unavailable."))

if not df_h.empty:
    with st.expander("Hurst R/S details"):
        st.dataframe(df_h, use_container_width=True)
        fig3 = plt.figure(figsize=(6, 4))
        plt.plot(np.log(df_h["chunk"]), np.log(df_h["RS_mean"] + EPS), marker="o")
        plt.title("Hurst (R/S): log(R/S) vs log(chunk)")
        plt.xlabel("log(chunk size)")
        plt.ylabel("log(R/S)")
        plt.tight_layout()
        st.pyplot(fig3)

# Download results
st.divider()
st.subheader("Download fitted + forecast series")

out_df = pd.DataFrame({
    "Date": list(dates) + list(fwd_dates),
    "Actual": list(y) + [np.nan] * len(fwd_dates),
    "WaveFit": list(yhat) + [np.nan] * len(fwd_dates),
    "WaveForecast": [np.nan] * len(dates) + list(y_fwd),
})
csv_out = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_out, file_name="wave_fit_forecast.csv", mime="text/csv")
