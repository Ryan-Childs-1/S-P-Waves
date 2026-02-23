# app.py
# Commodity Waves: Walk-Forward "Consistent Wave" Finder + Hurst/Mean Reversion
#
# NEW (Comprehensive upgrade):
# ✅ Finds a SIMPLER, CONSISTENT wave formula by OPTIMIZING:
#    - lookback window length (L)
#    - number of harmonics (H)
#    - forecast horizon (K)
#   using WALK-FORWARD backtesting across the chosen timeframe.
#
# ✅ Outputs statistics you need to understand performance:
#    - Out-of-sample RMSE / MAE / MAPE (on price)
#    - Directional accuracy (hit rate of sign changes)
#    - R² on out-of-sample predictions
#    - Bias (mean error), error volatility
#    - Model complexity score (AIC-like + penalty on harmonics)
#    - Stability metrics across folds (std of errors, coef drift summary)
#
# ✅ Robust parsing (CSV/TSV/whitespace), numeric coercion, NaN cleaning
# ✅ Safe harmonic caps + condition-number warnings
# ✅ Shows best (L,H) found + the final fitted formula on the last window,
#    then projects forward K steps.
#
# Requirements:
#   streamlit
#   numpy
#   pandas
#   matplotlib
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
st.set_page_config(page_title="Commodity Waves: Consistent Wave Finder", layout="wide")

DEFAULT_CSV_NAME = "SPX.csv"
EPS = 1e-12

IGNORE_LOCAL_CSV_NAMES = {
    "wave_fit_forecast.csv",
    "results.csv",
    "output.csv",
}

# Guardrails
MAX_TOTAL_ROWS_FOR_BACKTEST = 200_000   # prevent huge compute
MAX_BACKTEST_FOLDS = 600               # cap walk-forward evaluations
MAX_HARMONICS_UI = 200                 # UI ceiling
COND_WARN = 1e10                       # design matrix condition warning


# -----------------------------
# Utilities
# -----------------------------
def human_int(n: int) -> str:
    return f"{int(n):,}"

def safe_float(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")

def list_local_csvs(folder: str = ".") -> List[str]:
    out = []
    try:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".csv") and fn not in IGNORE_LOCAL_CSV_NAMES:
                out.append(fn)
    except Exception:
        return []
    out = sorted(out, key=lambda s: s.lower())
    if DEFAULT_CSV_NAME in out:
        out.remove(DEFAULT_CSV_NAME)
        out.insert(0, DEFAULT_CSV_NAME)
    return out

@st.cache_data(show_spinner=False)
def read_local_csv_bytes(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()

def _read_table_sniff(file_bytes: bytes) -> pd.DataFrame:
    raw = file_bytes.decode("utf-8", errors="replace")
    # 1) sniff delimiter
    try:
        return pd.read_csv(io.StringIO(raw), sep=None, engine="python")
    except Exception:
        pass
    # 2) tab
    try:
        return pd.read_csv(io.StringIO(raw), sep="\t")
    except Exception:
        pass
    # 3) comma
    try:
        return pd.read_csv(io.StringIO(raw), sep=",")
    except Exception:
        pass
    # 4) whitespace
    return pd.read_csv(io.StringIO(raw), delim_whitespace=True)

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    df = _read_table_sniff(file_bytes)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "datetime", "timestamp", "time"):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
    if date_col is None:
        raise ValueError("Could not find a Date column (Date/Datetime/Timestamp/Time or contains 'date').")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    df = df.set_index(date_col)
    df = df.dropna(axis=1, how="all")
    return df

def robust_column_choices(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols

def infer_forward_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "D"
    diffs = np.diff(index.values).astype("timedelta64[D]").astype(int)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return "D"
    med = int(np.median(diffs))
    return "B" if med <= 2 else "D"


# -----------------------------
# Wave model (Fourier + trend)
# y0(t) = a0 + a1*t + Σ_{k=1..H}( b_k cos(2πk t/T) + c_k sin(2πk t/T) )
# y(t)  = mean + std * y0(t)
# -----------------------------
@dataclass
class WaveFitResult:
    T: float
    harmonics: int
    coef: np.ndarray
    y_mean: float
    y_std: float
    rmse: float
    r2: float
    cond: float
    n: int

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

def safe_harmonics_cap(n: int, H: int) -> int:
    # Keep parameters well below n to avoid singular fits.
    # params = 2 + 2H. Rule: params <= n/4  => H <= (n/4 - 2)/2
    H = int(max(1, H))
    maxH = int(max(1, (n // 4 - 2) // 2))
    return int(min(H, maxH))

def fit_wave_formula(y: np.ndarray, harmonics: int) -> WaveFitResult:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        raise ValueError("Need ~20+ finite points to fit.")

    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + EPS)
    if not np.isfinite(y_mean) or not np.isfinite(y_std) or y_std < 1e-15:
        raise ValueError("Series not suitable (mean/std invalid or near-constant).")

    y0 = (y - y_mean) / y_std

    t = np.arange(n, dtype=np.float64)
    T = float(max(2.0, n - 1))
    H = safe_harmonics_cap(n, int(harmonics))

    X = build_design_matrix(t, T=T, H=H)

    try:
        cond = float(np.linalg.cond(X))
    except Exception:
        cond = float("nan")

    coef, *_ = np.linalg.lstsq(X, y0, rcond=None)
    yhat0 = X @ coef
    resid = y0 - yhat0

    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y0 - y0.mean()) ** 2) + EPS)
    r2 = float(1.0 - ss_res / ss_tot)

    return WaveFitResult(
        T=T, harmonics=H, coef=coef.astype(np.float64),
        y_mean=y_mean, y_std=y_std, rmse=rmse, r2=r2,
        cond=cond, n=n
    )

def eval_wave_formula(fit: WaveFitResult, n_points: int, start_t: float = 0.0) -> np.ndarray:
    t = start_t + np.arange(int(n_points), dtype=np.float64)
    X = build_design_matrix(t, T=fit.T, H=fit.harmonics)
    y0 = X @ fit.coef
    return (y0 * fit.y_std + fit.y_mean).astype(np.float64)

def wave_formula_text(fit: WaveFitResult, max_terms: int = 25) -> str:
    a0 = fit.coef[0]
    a1 = fit.coef[1]
    H = fit.harmonics
    bs = fit.coef[2:2 + H]
    cs = fit.coef[2 + H:2 + 2 * H]
    show = min(H, int(max_terms))

    lines = []
    lines.append("y(t) = mean + std * [ a0 + a1*t + Σ_{k=1..H}( b_k cos(2πk t / T) + c_k sin(2πk t / T) ) ]")
    lines.append("")
    lines.append(f"mean = {fit.y_mean:.6g}")
    lines.append(f"std  = {fit.y_std:.6g}")
    lines.append(f"T    = {fit.T:.6g}")
    lines.append(f"H    = {fit.harmonics}")
    lines.append(f"cond(X) ≈ {fit.cond:.3g}")
    lines.append("")
    lines.append(f"a0 = {a0:.6g}")
    lines.append(f"a1 = {a1:.6g}")
    for k in range(1, show + 1):
        lines.append(f"b{k} = {bs[k-1]:.6g}    c{k} = {cs[k-1]:.6g}")
    if H > show:
        lines.append(f"... ({H - show} more harmonics not shown)")
    return "\n".join(lines)


# -----------------------------
# Forecast evaluation / selection
# -----------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan"),
                "bias": float("nan"), "err_std": float("nan"), "dir_acc": float("nan")}

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), EPS)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2) + EPS)
    r2 = float(1.0 - ss_res / ss_tot)

    bias = float(np.mean(err))
    err_std = float(np.std(err) + EPS)

    # Directional accuracy (sign of next-step change) on overlapping region
    if y_true.size >= 2:
        dy_true = np.diff(y_true)
        dy_pred = np.diff(y_pred)
        dir_acc = float(np.mean((np.sign(dy_true) == np.sign(dy_pred)).astype(float)))
    else:
        dir_acc = float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "bias": bias, "err_std": err_std, "dir_acc": dir_acc}

def complexity_penalty(n: int, H: int) -> float:
    # Params = 2 + 2H
    p = 2 + 2 * int(H)
    # Mild penalty increasing with params, scaled for comparability
    return float(p * math.log(max(n, 2)))

def score_model(oos_rmse: float, n: int, H: int, lam: float) -> float:
    """
    Lower is better.
    score = log(RMSE) + lam * (penalty / n)
    """
    oos_rmse = max(float(oos_rmse), 1e-12)
    pen = complexity_penalty(n=n, H=H) / max(n, 1)
    return float(math.log(oos_rmse) + float(lam) * pen)

def walk_forward_backtest(
    y: np.ndarray,
    lookback: int,
    harmonics: int,
    horizon: int,
    stride: int,
    max_folds: int,
) -> Dict[str, Any]:
    """
    Walk-forward:
      For t0 in [lookback .. N-horizon] stepping by stride:
        fit on y[t0-lookback : t0]
        predict next horizon points
        compare to y[t0 : t0+horizon]
    Returns aggregated out-of-sample metrics and stability stats.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    N = len(y)
    L = int(lookback)
    H = int(harmonics)
    K = int(horizon)
    stride = int(max(1, stride))
    max_folds = int(max(1, max_folds))

    if N < L + K + 5:
        return {"ok": False, "reason": "Not enough points for this lookback+horizon."}

    preds = []
    trues = []
    conds = []
    rmses_fold = []
    coef_summ = []

    folds = 0
    for t0 in range(L, N - K + 1, stride):
        y_train = y[t0 - L:t0]
        y_true = y[t0:t0 + K]

        try:
            fit = fit_wave_formula(y_train, harmonics=H)
            y_pred = eval_wave_formula(fit, n_points=K, start_t=float(L))  # continue time after train window
        except Exception:
            continue

        if not (np.all(np.isfinite(y_pred)) and np.all(np.isfinite(y_true))):
            continue

        preds.append(y_pred)
        trues.append(y_true)
        conds.append(fit.cond)
        rmses_fold.append(float(np.sqrt(np.mean((y_pred - y_true) ** 2))))
        # coefficient drift proxy: norm of coef (stable-ish if consistent)
        coef_summ.append(float(np.linalg.norm(fit.coef)))

        folds += 1
        if folds >= max_folds:
            break

    if folds < 5:
        return {"ok": False, "reason": f"Too few successful folds ({folds}). Try smaller H or different L."}

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)
    m = _metrics(T, P)

    return {
        "ok": True,
        "folds": folds,
        "oos_rmse": m["rmse"],
        "oos_mae": m["mae"],
        "oos_mape": m["mape"],
        "oos_r2": m["r2"],
        "dir_acc": m["dir_acc"],
        "bias": m["bias"],
        "err_std": m["err_std"],
        "rmse_fold_mean": float(np.mean(rmses_fold)),
        "rmse_fold_std": float(np.std(rmses_fold) + EPS),
        "cond_median": float(np.nanmedian(np.asarray(conds, dtype=np.float64))),
        "cond_p95": float(np.nanpercentile(np.asarray(conds, dtype=np.float64), 95)),
        "coef_norm_mean": float(np.mean(coef_summ)),
        "coef_norm_std": float(np.std(coef_summ) + EPS),
    }


# -----------------------------
# Hurst / Mean Reversion
# -----------------------------
def hurst_rs(series: np.ndarray, min_chunk: int = 16, max_chunk: int = 512) -> Tuple[float, pd.DataFrame]:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < max(64, min_chunk * 4):
        return float("nan"), pd.DataFrame()

    sizes = []
    s = int(min_chunk)
    while s <= min(int(max_chunk), N // 2):
        sizes.append(s)
        s *= 2

    rows = []
    for n in sizes:
        m = N // n
        if m < 2:
            continue
        rs_vals = []
        for i in range(m):
            seg = x[i * n:(i + 1) * n]
            seg = seg - seg.mean()
            z = np.cumsum(seg)
            R = z.max() - z.min()
            S = seg.std() + EPS
            rs_vals.append(R / S)
        rows.append({"chunk": n, "RS_mean": float(np.mean(rs_vals))})

    df = pd.DataFrame(rows)
    if df.empty or df["RS_mean"].le(0).all():
        return float("nan"), df

    lx = np.log(df["chunk"].values.astype(np.float64))
    ly = np.log(df["RS_mean"].values.astype(np.float64) + EPS)
    A = np.vstack([np.ones_like(lx), lx]).T
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(beta[1]), df

def mean_reversion_stats(price: np.ndarray) -> Dict[str, Any]:
    p = np.asarray(price, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 50:
        return {"ok": False, "reason": "Not enough points for mean-reversion stats (need ~50+)."}

    logp = np.log(np.maximum(p, EPS))
    r = np.diff(logp)
    if len(r) < 3:
        return {"ok": False, "reason": "Not enough points for returns regression."}

    r1 = r[1:]
    r0 = r[:-1]
    X = np.vstack([np.ones_like(r0), r0]).T
    beta, *_ = np.linalg.lstsq(X, r1, rcond=None)
    phi_ret = float(beta[1])

    x = logp - logp.mean()
    x1 = x[1:]
    x0 = x[:-1]
    Xp = np.vstack([np.ones_like(x0), x0]).T
    bp, *_ = np.linalg.lstsq(Xp, x1, rcond=None)
    b_p = float(bp[1])

    half_life = float("nan")
    if 0.0 < b_p < 1.0:
        half_life = float(-math.log(2.0) / math.log(b_p))

    return {
        "ok": True,
        "phi_returns": phi_ret,
        "b_logprice": b_p,
        "half_life_steps": half_life,
        "interpretation": (
            "phi_returns < 0 suggests short-term mean reversion (oscillatory returns); "
            "phi_returns > 0 suggests short-term momentum in returns. "
            "b_logprice closer to 1 implies slow mean reversion in log-price deviations."
        ),
    }


# -----------------------------
# UI
# -----------------------------
st.title("Consistent Wave Formula Finder (Lookback + Harmonics) + Forecast")

st.markdown(
    "Goal: find a **simpler** wave formula that **predicts forward consistently** by optimizing:\n"
    "- **Lookback window (L)** used to fit the wave\n"
    "- **# harmonics (H)** (complexity)\n"
    "- **Forecast horizon (K)** (what you want to predict)\n\n"
    "We use **walk-forward backtesting** to measure true out-of-sample performance.\n"
)

with st.sidebar:
    st.header("1) Data source")
    local_csvs = list_local_csvs(".")
    source = st.radio("CSV source", ["Pick CSV from app folder", "Upload CSV"], index=0)
    csv_bytes = None
    filename = "data.csv"

    if source == "Pick CSV from app folder":
        if not local_csvs:
            st.error("No .csv files found next to app.py. Add CSVs or choose Upload.")
        else:
            chosen = st.selectbox("Choose local CSV", local_csvs, index=0)
            csv_bytes = read_local_csv_bytes(chosen)
            filename = chosen
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            csv_bytes = up.getvalue()
            filename = getattr(up, "name", "uploaded.csv")

    st.divider()
    st.header("2) Column + timeframe")
    dropna_mode = st.selectbox("Cleaning", ["Drop non-numeric / NaNs (recommended)", "Keep (may fail)"], index=0)

    st.divider()
    st.header("3) Backtest optimization")
    # Search grids
    lookback_choices = st.multiselect(
        "Lookback candidates (L)",
        options=[30, 45, 60, 90, 126, 180, 252, 378, 504, 756, 1008],
        default=[126, 252, 504]
    )
    harmonics_choices = st.multiselect(
        "Harmonics candidates (H)",
        options=[1, 2, 3, 5, 8, 13, 21, 30, 40, 60],
        default=[3, 5, 8, 13, 21]
    )
    horizon = st.select_slider("Forecast horizon (K steps)", options=[1, 3, 5, 10, 21, 42, 63, 126], value=21)
    stride = st.select_slider("Walk-forward stride", options=[1, 3, 5, 10, 21], value=5)

    st.divider()
    st.header("4) Scoring + caps")
    lam = st.slider("Complexity penalty λ (higher = simpler)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
    max_folds = st.select_slider("Max folds (speed cap)", options=[50, 100, 200, 400, 600], value=200)
    max_rows = st.select_slider("Max rows used (speed cap)", options=[5_000, 10_000, 25_000, 50_000, 100_000], value=25_000)

    st.divider()
    st.header("5) Hurst settings")
    hurst_min = st.select_slider("Min chunk", options=[8, 16, 32, 64], value=16)
    hurst_max = st.select_slider("Max chunk", options=[128, 256, 512, 1024, 2048], value=512)

    show_debug = st.checkbox("Show debug details", value=False)

if csv_bytes is None:
    st.stop()

# Load & parse
try:
    df = parse_csv(csv_bytes)
except Exception as e:
    st.error(f"Failed to read {filename}: {e}")
    st.stop()

cols = robust_column_choices(df)
if not cols:
    st.error("No numeric-like columns found.")
    st.stop()

min_date = df.index.min().date()
max_date = df.index.max().date()

st.caption(
    f"Loaded: **{filename}** | Rows: **{human_int(len(df))}** | Range: **{min_date} → {max_date}**"
)

# Column selection + timeframe
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

before_rows = len(sub)
sub[col] = pd.to_numeric(sub[col], errors="coerce")
if dropna_mode.startswith("Drop"):
    sub = sub.dropna(subset=[col])
dropped = before_rows - len(sub)

if len(sub) < 200:
    st.warning("Optimization works best with more data. Consider a larger timeframe (200+ points).")

if dropped > 0:
    st.info(f"Dropped {dropped} rows with missing/non-numeric '{col}'.")

dates = sub.index
y_full = sub[col].astype(float).values

# Cap rows for speed (take most recent max_rows to match typical trading use)
if len(y_full) > int(max_rows):
    y_full = y_full[-int(max_rows):]
    dates = dates[-int(max_rows):]
    st.info(f"Speed cap: using most recent {human_int(len(y_full))} rows for optimization.")

# Plot series
st.subheader("Selected series")
fig0 = plt.figure(figsize=(10, 3))
plt.plot(dates, y_full)
plt.title(f"{filename} — {col}")
plt.xlabel("Date")
plt.ylabel(col)
plt.tight_layout()
st.pyplot(fig0)

# -----------------------------
# Optimization / backtest
# -----------------------------
st.divider()
st.subheader("Find best lookback + wave complexity (walk-forward)")

run_opt = st.button("Run optimization (walk-forward)", type="primary")

if "best_model" not in st.session_state:
    st.session_state["best_model"] = None
    st.session_state["opt_table"] = None

if run_opt:
    if not lookback_choices or not harmonics_choices:
        st.error("Pick at least one lookback and one harmonics candidate.")
    else:
        rows = []
        total = len(lookback_choices) * len(harmonics_choices)
        prog = st.progress(0.0)
        done = 0

        for L in lookback_choices:
            for Hcand in harmonics_choices:
                done += 1
                prog.progress(min(1.0, done / max(total, 1)))

                res = walk_forward_backtest(
                    y=y_full,
                    lookback=int(L),
                    harmonics=int(Hcand),
                    horizon=int(horizon),
                    stride=int(stride),
                    max_folds=int(min(max_folds, MAX_BACKTEST_FOLDS)),
                )
                if not res.get("ok"):
                    rows.append({
                        "lookback": int(L),
                        "harmonics": int(Hcand),
                        "folds": 0,
                        "oos_rmse": np.nan,
                        "oos_mae": np.nan,
                        "oos_mape_%": np.nan,
                        "oos_r2": np.nan,
                        "dir_acc": np.nan,
                        "rmse_fold_std": np.nan,
                        "cond_p95": np.nan,
                        "score": np.nan,
                        "reason": res.get("reason", "failed"),
                    })
                    continue

                sc = score_model(res["oos_rmse"], n=int(L), H=int(Hcand), lam=float(lam))
                rows.append({
                    "lookback": int(L),
                    "harmonics": int(Hcand),
                    "folds": int(res["folds"]),
                    "oos_rmse": float(res["oos_rmse"]),
                    "oos_mae": float(res["oos_mae"]),
                    "oos_mape_%": float(res["oos_mape"]),
                    "oos_r2": float(res["oos_r2"]),
                    "dir_acc": float(res["dir_acc"]),
                    "bias": float(res["bias"]),
                    "err_std": float(res["err_std"]),
                    "rmse_fold_std": float(res["rmse_fold_std"]),
                    "cond_p95": float(res["cond_p95"]),
                    "coef_norm_std": float(res["coef_norm_std"]),
                    "score": float(sc),
                    "reason": "",
                })

        opt_df = pd.DataFrame(rows)
        opt_df = opt_df.sort_values(["score", "oos_rmse"], ascending=[True, True], na_position="last").reset_index(drop=True)

        st.session_state["opt_table"] = opt_df

        # pick best valid row
        best_row = opt_df.dropna(subset=["score"]).head(1)
        if best_row.empty:
            st.session_state["best_model"] = None
            st.error("No valid (lookback, harmonics) combo succeeded. Try smaller H or smaller horizon.")
        else:
            br = best_row.iloc[0].to_dict()
            st.session_state["best_model"] = br
            st.success(
                f"Best: lookback={int(br['lookback'])}, harmonics={int(br['harmonics'])}, "
                f"OOS_RMSE={br['oos_rmse']:.4g}, dir_acc={br['dir_acc']:.2%}, score={br['score']:.4g}"
            )

opt_df = st.session_state.get("opt_table", None)
best = st.session_state.get("best_model", None)

if opt_df is not None:
    with st.expander("Optimization results table"):
        st.dataframe(opt_df, use_container_width=True)

    # quick viz: RMSE vs lookback for each harmonics
    try:
        fig_rmse = plt.figure(figsize=(10, 4))
        for Hcand in sorted(set(opt_df["harmonics"].dropna().astype(int).tolist())):
            dfh = opt_df[(opt_df["harmonics"] == Hcand) & np.isfinite(opt_df["oos_rmse"].values)]
            if len(dfh) == 0:
                continue
            plt.plot(dfh["lookback"], dfh["oos_rmse"], marker="o", label=f"H={Hcand}")
        plt.title("Out-of-sample RMSE by lookback (per harmonics)")
        plt.xlabel("Lookback (L)")
        plt.ylabel("OOS RMSE")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig_rmse)
    except Exception:
        pass

if best is None:
    st.info("Run optimization to select a consistent lookback + wave complexity.")
    st.stop()

# -----------------------------
# Fit final model on most recent lookback window, then forecast forward
# -----------------------------
L_best = int(best["lookback"])
H_best = int(best["harmonics"])
K = int(horizon)

if len(y_full) < L_best + 5:
    st.error("Not enough points to fit the best model on the end of the series.")
    st.stop()

train_y = np.asarray(y_full[-L_best:], dtype=np.float64)
train_dates = dates[-L_best:]

fit_final = fit_wave_formula(train_y, harmonics=H_best)

# Forecast K steps forward
y_fit_in = eval_wave_formula(fit_final, n_points=L_best, start_t=0.0)
y_fwd = eval_wave_formula(fit_final, n_points=K, start_t=float(L_best))

freq = infer_forward_freq(dates)
fwd_dates = pd.date_range(dates.max(), periods=K + 1, freq=freq)[1:]

st.divider()
st.subheader("Best consistent wave model (final fit on last lookback window)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Best lookback (L)", str(L_best))
m2.metric("Best harmonics (H)", str(fit_final.harmonics))
m3.metric("OOS RMSE (avg)", f"{best['oos_rmse']:.4g}")
m4.metric("Directional acc", "NaN" if not np.isfinite(best["dir_acc"]) else f"{best['dir_acc']:.2%}")
m5.metric("cond(X) (final)", "NaN" if not np.isfinite(fit_final.cond) else f"{fit_final.cond:.2g}")

if np.isfinite(fit_final.cond) and fit_final.cond > COND_WARN:
    st.warning("Final fit is ill-conditioned. Reduce harmonics or shorten lookback.")

# Plot: last lookback window + its fit + forward projection
fig1 = plt.figure(figsize=(12, 4))
plt.plot(train_dates, train_y, label="Actual (lookback window)")
plt.plot(train_dates, y_fit_in, label="Wave fit (in-window)")
plt.plot(fwd_dates, y_fwd, label=f"Forward projection ({K} steps)")
plt.axvline(train_dates.max(), linestyle="--", linewidth=1)
plt.title(f"{filename} — {col}: Consistent wave fit (L={L_best}, H={fit_final.harmonics}) + forecast")
plt.xlabel("Date")
plt.ylabel(col)
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

with st.expander("Show final wave formula coefficients"):
    st.text(wave_formula_text(fit_final))

with st.expander("Explain what the optimizer is doing"):
    st.markdown(
        "- For each candidate **lookback L** and **harmonics H**, we repeatedly:\n"
        "  1) fit the wave on the last **L** points of a rolling window\n"
        "  2) predict the next **K** points\n"
        "  3) score the out-of-sample errors\n"
        "- We then choose the model with the best **score**:\n"
        "  - lower out-of-sample RMSE is better\n"
        "  - higher λ penalizes complexity (more harmonics) so formulas simplify\n"
    )

# -----------------------------
# Hurst + Mean Reversion on the full selected series
# -----------------------------
st.divider()
st.subheader("Hurst exponent + mean reversion diagnostics (full selected timeframe)")

hurst_mode = st.radio("Hurst series", ["Log price", "Log returns"], horizontal=True, index=0, key="hurst_mode")
p = np.asarray(y_full, dtype=np.float64)
logp = np.log(np.maximum(p, EPS))
series = logp if hurst_mode == "Log price" else np.diff(logp)

Hurst, df_h = hurst_rs(series, min_chunk=int(hurst_min), max_chunk=int(hurst_max))
mr = mean_reversion_stats(p)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Hurst (R/S)", "NaN" if not np.isfinite(Hurst) else f"{Hurst:.3f}")
h2.metric(
    "Interpretation",
    "—" if not np.isfinite(Hurst)
    else ("Mean reverting" if Hurst < 0.5 else ("Random-walk-ish" if abs(Hurst - 0.5) < 0.05 else "Trending"))
)
if mr.get("ok"):
    h3.metric("AR(1) phi on returns", f"{mr['phi_returns']:.3f}")
    h4.metric("Half-life (log-price dev)", "NaN" if not np.isfinite(mr["half_life_steps"]) else f"{mr['half_life_steps']:.1f} steps")
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
        fig2 = plt.figure(figsize=(6, 4))
        plt.plot(np.log(df_h["chunk"]), np.log(df_h["RS_mean"] + EPS), marker="o")
        plt.title("Hurst (R/S): log(R/S) vs log(chunk)")
        plt.xlabel("log(chunk size)")
        plt.ylabel("log(R/S)")
        plt.tight_layout()
        st.pyplot(fig2)

# -----------------------------
# Downloads
# -----------------------------
st.divider()
st.subheader("Download: optimization + forecast")

# Forecast output: stitch last window + future
out_df = pd.DataFrame({
    "Date": list(train_dates) + list(fwd_dates),
    "Actual": list(train_y) + [np.nan] * len(fwd_dates),
    "WaveFit": list(y_fit_in) + [np.nan] * len(fwd_dates),
    "WaveForecast": [np.nan] * len(train_dates) + list(y_fwd),
})
csv_out = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast CSV", data=csv_out, file_name="wave_forecast_best.csv", mime="text/csv")

if opt_df is not None:
    opt_out = opt_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download optimization table CSV", data=opt_out, file_name="wave_optimization_table.csv", mime="text/csv")
