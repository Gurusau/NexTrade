"""


NexTrade — Next-Day Stock Prediction
A Multivariate Deep Learning System for Next-Day Stock closing Price Prediction


--------------------------------

Done By -
Saurabh kumar ( 2203322 )
Varun ( 2203137 )
Imtiyaz Hussain ( 2203113 ) 


"""

import os
import io
from typing import Dict, List, Optional, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title=" NexTrade ", page_icon="📈", layout="wide")
st.title("📈 NexTrade — Stock Prediction ")

DATA_FOLDER = "data"
SEQ_LEN_DEFAULT = 10
STALE_WARNING_DAYS_DEFAULT = 3

# ---------------------------
# Quick tickers for sidebar (friendly display)
# ---------------------------
QUICK_TICKERS = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corp."),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com Inc."),
    ("TSLA", "Tesla Inc."),
    ("TCS.NS", "Tata Consultancy Services (NSE)"),
    ("RELIANCE.NS", "Reliance Industries (NSE)"),
    ("INFY.NS", "Infosys (NSE)"),
    ("HDFCBANK.NS", "HDFC Bank (NSE)"),
    ("BPCL.NS", "BPCL (NSE)")
]

# ---------------------------
# Minimal sidebar: show the 10 tickers and let user pick default quick-ticker
# ---------------------------
with st.sidebar:
    st.header("Quick picks")
    st.write("If you don't know a ticker code, pick from these common companies:")
    quick_labels = [f"{code} — {name}" for code, name in QUICK_TICKERS]
    quick_choice = st.radio("Choose a company (sidebar)", quick_labels, index=0)
    DEFAULT_QUICK_TICKER = quick_choice.split("—")[0].strip()

# ---------------------------
# Improved Yahoo downloader (cached)
# ---------------------------
@st.cache_data(ttl=60 * 60 * 2)
def download_from_yfinance(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    """
    Robust download wrapper around yfinance.

    Returns (df, info) where df is a DataFrame with columns Date, Close, Volume
    (or empty if it failed) and info is a diagnostic string explaining success
    or describing the reason for failure.
    """
    ticker = str(ticker).strip()
    if ticker == "":
        return pd.DataFrame(), "Empty ticker provided."

    attempts_info = []
    strategies = [
        {"fn": lambda: yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False),
         "desc": "yf.download(auto_adjust=True)"},
        {"fn": lambda: yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False),
         "desc": "yf.download(auto_adjust=False)"},
        {"fn": lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True),
         "desc": "Ticker.history(auto_adjust=True)"},
        {"fn": lambda: yf.Ticker(ticker).history(period=period, auto_adjust=False),
         "desc": "Ticker.history(auto_adjust=False)"},
        {"fn": lambda: yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False, threads=False),
         "desc": "yf.download(interval=1d, auto_adjust=True)"},
    ]

    last_exception_msg = None
    df_result = pd.DataFrame()

    for strat in strategies:
        try:
            df = strat["fn"]()
            if df is None:
                attempts_info.append(f"{strat['desc']}: returned None")
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame().T
            # If dates are index, reset and call column 'Date'
            if isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
                df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
            # Normalize column names
            cols_lower = [c.lower() if isinstance(c, str) else c for c in df.columns]
            if "close" not in cols_lower and "adj close" in cols_lower:
                df = df.rename(columns={c: "Close" for c in df.columns if isinstance(c, str) and c.lower() == "adj close"})
            # Map known columns
            cols_map = {}
            for c in df.columns:
                if isinstance(c, str):
                    cl = c.lower()
                    if cl == "close" or cl == "adj close":
                        cols_map[c] = "Close"
                    if cl == "volume":
                        cols_map[c] = "Volume"
                    if cl == "date" or cl == "index":
                        cols_map[c] = "Date"
            if cols_map:
                df = df.rename(columns=cols_map)
            # Ensure Date exists
            if "Date" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
                else:
                    date_candidates = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
                    if date_candidates:
                        df = df.rename(columns={date_candidates[0]: "Date"})
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            def _clean_num(s):
                return pd.to_numeric(s.astype(str).str.replace(r"[,\$\€\₹\s]", "", regex=True), errors="coerce")
            if "Close" in df.columns:
                df["Close"] = _clean_num(df["Close"])
            if "Volume" in df.columns:
                df["Volume"] = _clean_num(df["Volume"])
            else:
                df["Volume"] = 0.0
            df = df.dropna(subset=["Close"]).sort_values("Date" if "Date" in df.columns else df.columns[0]).reset_index(drop=True)
            if df.empty:
                attempts_info.append(f"{strat['desc']}: got empty after cleaning.")
                continue
            df_result = df[["Date", "Close", "Volume"]].copy()
            info = f"Success via {strat['desc']}"
            return df_result, info
        except Exception as e:
            last_exception_msg = str(e)
            attempts_info.append(f"{strat['desc']}: exception: {e}")
            continue

    info = " ; ".join(attempts_info)
    if last_exception_msg:
        info += " ; last_exc: " + last_exception_msg
    return pd.DataFrame(), f"No data from Yahoo for {ticker}. Attempts: {info}"

# ---------------------------
# Holiday & trading-day helpers
# ---------------------------
def load_holidays_from_df(df: pd.DataFrame) -> List[pd.Timestamp]:
    df2 = df.copy()
    date_cols = [c for c in df2.columns if "date" in str(c).lower()]
    if not date_cols:
        return []
    dc = date_cols[0]
    df2 = df2.rename(columns={dc: "date"})
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce", dayfirst=False)
    if df2["date"].isna().sum() > 0 and df2["date"].notna().sum() > 0:
        alt = pd.to_datetime(df2["date"].astype(str), errors="coerce", dayfirst=True)
        if alt.isna().sum() < df2["date"].isna().sum():
            df2["date"] = alt
    hs = sorted(pd.to_datetime(df2["date"].dropna()).dt.normalize().unique())
    return list(pd.to_datetime(hs))


def next_trading_day(start_date: pd.Timestamp, holidays: Optional[List[pd.Timestamp]] = None) -> pd.Timestamp:
    if start_date is None:
        return pd.Timestamp.today().normalize()
    d = pd.to_datetime(start_date).normalize() + pd.Timedelta(days=1)
    hol_set = set(pd.to_datetime(holidays).normalize()) if holidays else set()
    while d.weekday() >= 5 or d in hol_set:
        d += pd.Timedelta(days=1)
    return d

# ---------------------------
# CSV normalization & load helpers
# ---------------------------
@st.cache_data(ttl=60 * 60 * 2)
def list_data_folder_tickers(folder: str = DATA_FOLDER) -> List[str]:
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    tickers = [os.path.splitext(f)[0] for f in files]
    return sorted(tickers)


@st.cache_data(ttl=60 * 60 * 2)
def load_csv_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_csv_df(df)
    return df


@st.cache_data(ttl=60 * 60 * 2)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = _normalize_csv_df(df)
    return df


def _choose_close_column(columns: List[str]) -> Optional[str]:
    cols = [c for c in columns]
    if "close" in cols:
        return "close"
    for name in ("adj close", "adjusted close", "adj_close", "adjusted_close", "adjclose", "adjustedclose"):
        if name in cols:
            return name
    close_candidates = [c for c in cols if "close" in c]
    non_prev = [c for c in close_candidates if ("prev" not in c and "previous" not in c)]
    if non_prev:
        return non_prev[0]
    if close_candidates:
        return close_candidates[0]
    return None


def _normalize_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        if isinstance(c, str):
            c2 = c.strip().replace("\ufeff", "").replace("\xa0", " ").lower()
            new_cols.append(c2)
        else:
            new_cols.append(str(c))
    df.columns = new_cols

    date_candidates = [c for c in df.columns if ("date" in c) or ("time" in c) or ("timestamp" in c)]
    if not date_candidates:
        raise ValueError("CSV must contain a Date column (like 'Date' or 'date').")
    date_col = date_candidates[0]
    df = df.rename(columns={date_col: "Date"})

    close_choice = _choose_close_column(list(df.columns))
    if close_choice:
        df = df.rename(columns={close_choice: "Close"})
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[-1]: "Close"})
        else:
            raise ValueError("CSV must contain a Close (or Adj Close) numeric column.")

    vol_candidates = [c for c in df.columns if "vol" in c]
    if vol_candidates:
        df = df.rename(columns={vol_candidates[0]: "Volume"})
    else:
        df["Volume"] = 0.0

    def clean_numeric_series(s: pd.Series) -> pd.Series:
        s2 = s.astype(str).str.strip()
        s2 = s2.str.replace(r"[,\$\€\₹\s]", "", regex=True)
        s2 = s2.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        s2 = s2.str.replace("%", "", regex=False)
        return pd.to_numeric(s2, errors="coerce")

    df["Close"] = clean_numeric_series(df["Close"])
    df["Volume"] = clean_numeric_series(df["Volume"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    nat_count = df["Date"].isna().sum()
    if nat_count > 0 and nat_count < len(df):
        alt = pd.to_datetime(df["Date"].astype(str), errors="coerce", dayfirst=True)
        if alt.isna().sum() < nat_count:
            df["Date"] = alt
    if df["Date"].isna().all():
        raise ValueError("Could not parse any dates in the Date column. Try a different CSV format.")

    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df

# ---------------------------
# features / model utilities
# ---------------------------
def rows_for_period(period: str) -> int:
    if not isinstance(period, str) or period.strip() == "":
        return 200
    p = period.strip().lower()
    try:
        if p.endswith("y"):
            years = float(p[:-1])
            return max(1, int(years * 200))
        if p.endswith("m"):
            months = float(p[:-1])
            return max(1, int(months * 21))
        return int(float(p))
    except Exception:
        return 200


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns and "close" in df.columns:
        df["Close"] = df["close"]
    if "Volume" not in df.columns and "volume" in df.columns:
        df["Volume"] = df["volume"]

    if "Close" not in df.columns:
        raise ValueError("Close column missing in data")

    if not pd.api.types.is_numeric_dtype(df["Close"]):
        df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(r"[,\$\€\₹\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True), errors="coerce")

    if "Volume" in df.columns and not pd.api.types.is_numeric_dtype(df["Volume"]):
        df["Volume"] = pd.to_numeric(df["Volume"].astype(str).str.replace(r"[,\$\€\₹\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True), errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    df["ret_1d"] = df["Close"].pct_change()
    for w in [5, 10]:
        df[f"ma_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"vol_{w}"] = df["ret_1d"].rolling(window=w).std()
        df[f"mom_{w}"] = df["Close"].pct_change(periods=w)
    df["volum"] = df["Volume"]
    df = df.dropna().reset_index(drop=True)
    return df


def create_sequences_multivariate(feature_matrix: np.ndarray, target_array: np.ndarray, seq_len: int):
    X, y = [], []
    N = feature_matrix.shape[0]
    if N < seq_len:
        return np.empty((0, seq_len, feature_matrix.shape[1])), np.empty((0,))
    for i in range(0, N - seq_len + 1):
        seq = feature_matrix[i: i + seq_len]
        targ = target_array[i + seq_len - 1]
        X.append(seq)
        y.append(targ)
    return np.array(X), np.array(y)


def build_cnn_lstm_multivariate(seq_len: int, n_features: int):
    model = models.Sequential([
        layers.Conv1D(filters=24, kernel_size=3, padding='causal', activation='relu', input_shape=(seq_len, n_features)),
        layers.BatchNormalization(),
        layers.LSTM(48, return_sequences=True),
        layers.Dropout(0.15),
        layers.LSTM(24, return_sequences=False),
        layers.Dropout(0.15),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------------
# duplicate-date resolution
# ---------------------------
def resolve_duplicate_date_rows(df: pd.DataFrame, date_col: str = "Date", close_col: str = "Close", strategy: str = "last"):
    df = df.copy()
    df["_date_only"] = pd.to_datetime(df[date_col]).dt.normalize()
    grouped = []
    for d, g in df.groupby("_date_only"):
        if len(g) == 1:
            grouped.append(g)
            continue
        if strategy == "last":
            grouped.append(g.tail(1))
        elif strategy == "latest_time":
            g = g.copy()
            g["__ts"] = pd.to_datetime(g[date_col], errors='coerce')
            g = g.sort_values("__ts").drop(columns="__ts")
            grouped.append(g.tail(1))
        elif strategy == "max_close":
            idx = g[close_col].idxmax()
            grouped.append(g.loc[[idx]])
        elif strategy == "min_close":
            idx = g[close_col].idxmin()
            grouped.append(g.loc[[idx]])
        else:
            grouped.append(g.tail(1))
    out = pd.concat(grouped, axis=0).sort_values(by=date_col).drop(columns=["_date_only"]).reset_index(drop=True)
    return out

# ---------------------------
# pipeline: supports source = 'local' | 'upload' | 'yahoo'
# ---------------------------
def run_pipeline_for_ticker(ticker: str = "", period: str = "1y", raw_df: Optional[pd.DataFrame] = None,
                            dup_strategy: str = "last", source: str = "local") -> Dict[str, Any]:
    ticker = str(ticker).strip().upper() if ticker else ""
    raw = None

    # load according to source
    if source == "yahoo":
        if ticker == "":
            return {"error": "empty_ticker"}
        raw, info = download_from_yfinance(ticker, period=period)
        if raw.empty:
            return {"error": "no_data", "message": info}
    elif source == "upload":
        if raw_df is None:
            return {"error": "no_data", "message": "No uploaded DataFrame provided."}
        try:
            raw = raw_df.copy()
            if "Date" not in raw.columns or "Close" not in raw.columns:
                raw = _normalize_csv_df(raw)
            else:
                raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
                if raw["Date"].isna().sum() > 0:
                    raw = _normalize_csv_df(raw)
        except Exception as e:
            return {"error": "no_data", "message": f"Uploaded CSV could not be interpreted: {e}"}
    else:  # local
        if ticker == "":
            return {"error": "empty_ticker"}
        path_candidates = [os.path.join(DATA_FOLDER, f"{ticker}.csv"), os.path.join(DATA_FOLDER, f"{ticker.lower()}.csv")]
        for p in path_candidates:
            if os.path.exists(p):
                try:
                    raw = load_csv_from_path(p)
                    break
                except Exception as e:
                    return {"error": "no_data", "message": f"Failed reading CSV {p}: {e}"}
        if raw is None:
            return {"error": "no_data", "message": f"No CSV found for {ticker} in {DATA_FOLDER}/"}

    # resolve duplicates
    try:
        raw = resolve_duplicate_date_rows(raw, date_col="Date", close_col="Close", strategy=dup_strategy)
    except Exception:
        pass

    # deterministic latest-date selection
    try:
        latest_date = raw["Date"].max()
        latest_rows = raw[raw["Date"] == latest_date]
        if latest_rows.empty:
            latest_rows = raw[pd.to_datetime(raw["Date"]).dt.normalize() == pd.to_datetime(latest_date).normalize()]
        if latest_rows.empty:
            latest_idx = raw["Date"].idxmax()
            raw_last_row = raw.loc[[latest_idx]].iloc[0]
        else:
            raw_last_row = latest_rows.tail(1).iloc[0]

        if "Close" not in raw_last_row:
            raise KeyError("Close column not present in latest row")
        raw_last_close = float(raw_last_row["Close"])
        raw_last_date = pd.to_datetime(raw_last_row["Date"])
    except Exception:
        raw_sorted = raw.sort_values("Date").reset_index(drop=True)
        raw_last_row = raw_sorted.iloc[-1]
        raw_last_close = float(raw_last_row["Close"])
        raw_last_date = pd.to_datetime(raw_last_row["Date"])

    # prepare features
    df_for_features = raw.sort_values("Date")[["Date", "Close", "Volume"]].copy()
    try:
        df_feat_full = add_features(df_for_features.rename(columns={"Date": "Date", "Close": "Close", "Volume": "Volume"}))
    except Exception as e:
        return {"error": "processing_error", "message": f"Feature creation failed: {e}"}

    df_feat = df_feat_full.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    df_feat["ret_next"] = df_feat["close"].shift(-1) / df_feat["close"] - 1.0
    df_feat = df_feat.dropna().reset_index(drop=True)

    if len(df_feat) < 80:
        return {"error": "not_enough_data", "n_rows": len(df_feat)}

    feature_cols = ["ret_1d", "ma_5", "ma_10", "vol_5", "vol_10", "mom_5", "mom_10", "volum"]
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    features = df_feat[feature_cols].values
    targets = df_feat["ret_next"].values

    SEQ_LEN = SEQ_LEN_DEFAULT
    X, y = create_sequences_multivariate(features, targets, SEQ_LEN)
    if X.shape[0] < 40:
        return {"error": "not_enough_data", "n_rows": len(df_feat)}

    N_seq = X.shape[0]
    split_idx = int(0.8 * N_seq)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_features = X.shape[2]
    X_train_flat = X_train.reshape(-1, n_features)
    scaler_X = StandardScaler()
    X_train_flat_scaled = scaler_X.fit_transform(X_train_flat)
    X_train_scaled = X_train_flat_scaled.reshape(X_train.shape)

    X_test_flat = X_test.reshape(-1, n_features)
    X_test_flat_scaled = scaler_X.transform(X_test_flat)
    X_test_scaled = X_test_flat_scaled.reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    model = build_cnn_lstm_multivariate(SEQ_LEN, n_features)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    model.fit(X_train_scaled, y_train_scaled, epochs=40, batch_size=16, validation_data=(X_test_scaled, y_test_scaled), callbacks=[early_stopping], verbose=0)

    pred_test_scaled = model.predict(X_test_scaled).flatten()
    pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()

    base_prices = np.array([df_feat["close"].values[i + SEQ_LEN - 1] for i in range(N_seq)])
    base_test = base_prices[split_idx:]
    pred_test_prices = base_test * (1.0 + pred_test)
    true_test_prices = base_test * (1.0 + y_test)

    mae_price = float(mean_absolute_error(true_test_prices, pred_test_prices))

    last_close = raw_last_close
    last_close_date = raw_last_date

    # last sequence -> next-day prediction
    last_seq = features[-SEQ_LEN:]
    last_seq_scaled = scaler_X.transform(last_seq.reshape(-1, n_features)).reshape(1, SEQ_LEN, n_features)
    next_ret_scaled = model.predict(last_seq_scaled).flatten()[0]
    next_ret = float(scaler_y.inverse_transform(np.array([[next_ret_scaled]]))[0, 0])
    next_pred_price = float(last_close * (1.0 + next_ret))

    # test dates mapping
    td_start = split_idx + SEQ_LEN
    td_end = td_start + len(base_test)
    if td_start < len(df_feat) and td_end <= len(df_feat):
        test_dates = pd.to_datetime(df_feat["date"].iloc[td_start:td_end].values)
    else:
        test_start_idx = SEQ_LEN - 1 + split_idx
        test_dates = pd.to_datetime(df_feat["date"].iloc[test_start_idx:].values)

    try:
        raw_last_row_serializable = raw_last_row.to_dict() if hasattr(raw_last_row, "to_dict") else dict(raw_last_row)
    except Exception:
        raw_last_row_serializable = {}
    raw_last_rows_serializable = raw.sort_values("Date").tail(5).to_dict(orient="records")

    return {
        "ticker": ticker or "UPLOADED",
        "last_close": last_close,
        "last_close_date": last_close_date,
        "pred_next_close": next_pred_price,
        "mae": mae_price,
        "df": df_feat,
        "test_dates": test_dates,
        "true_test_prices": true_test_prices,
        "pred_test_prices": pred_test_prices,
        "raw_last_row": raw_last_row_serializable,
        "raw_last_rows": raw_last_rows_serializable,
        "dup_strategy": dup_strategy,
        "source": source
    }

# ---------------------------
# Main UI (unchanged behavior, but fix days_old calculations to use date())
# ---------------------------
section = st.radio("Choose section", ["None", "Prediction", "Comparison", "Bulk Download"], index=0, horizontal=True)
st.markdown("---")

# ---------- Prediction ----------
if section == "Prediction":
    st.markdown("## Next Day Stock Prediction")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        data_source = st.selectbox("Data source", ["Local CSV (data/ folder)", "Upload CSV", "Yahoo Finance"], index=0)
        uploaded_file = None
        source_flag = "local"
        ticker = ""
        default_ticker = DEFAULT_QUICK_TICKER
        if data_source == "Local CSV (data/ folder)":
            available = list_data_folder_tickers()
            if available:
                ticker = st.selectbox("Choose ticker (from data/)", options=available, index=0)
            else:
                ticker = st.text_input("Enter ticker filename (place CSV in data/)", value=default_ticker)
            source_flag = "local"
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            source_flag = "upload"
        else:
            ticker = st.text_input("Yahoo ticker (e.g., AAPL, RELIANCE.NS)", value=default_ticker)
            source_flag = "yahoo"

    with col_b:
        period = st.selectbox("Select Period", ["1y", "2y", "5y"], index=0)

    run_btn = st.button(" 👁️‍🗨️ Show ")
    if run_btn:
        with st.spinner("Preparing data and training..."):
            if source_flag == "upload":
                if uploaded_file is None:
                    st.warning("Please upload a CSV file.")
                    st.stop()
                try:
                    raw_df = load_csv_from_upload(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to read uploaded CSV: {e}")
                    st.stop()
                result = run_pipeline_for_ticker(ticker="", period=period, raw_df=raw_df, dup_strategy="last", source="upload")
            elif source_flag == "yahoo":
                if not ticker:
                    st.error("Enter a Yahoo ticker.")
                    st.stop()
                result = run_pipeline_for_ticker(ticker=ticker.strip().upper(), period=period, raw_df=None, dup_strategy="last", source="yahoo")
            else:
                result = run_pipeline_for_ticker(ticker.strip().upper(), period, raw_df=None, dup_strategy="last", source="local")

        if result.get("error"):
            err = result.get("error")
            if err in ["no_data", "processing_error"]:
                st.error(result.get("message", f"No data for {ticker}"))
            elif err == "not_enough_data":
                st.warning(f"Not enough data — only {result.get('n_rows', '?')} rows available.")
            elif err == "empty_ticker":
                st.error("Ticker is empty.")
            else:
                st.error(f"Error: {err}")
        else:
            df = result["df"]
            display_n = rows_for_period(period)
            last_close = result["last_close"]
            last_close_date = result.get("last_close_date")
            pred = result["pred_next_close"]
            mae = result["mae"]

            # === FIX: compare dates (tz-naive vs tz-aware safe) ===
            if last_close_date is not None:
                try:
                    today_date = pd.Timestamp.today().normalize().date()
                    last_date_only = pd.to_datetime(last_close_date).date()
                    days_old = (today_date - last_date_only).days
                except Exception:
                    # fallback: use naive normalize subtraction but guarded
                    days_old = (pd.Timestamp.today().normalize() - pd.to_datetime(last_close_date).normalize()).days
                if days_old > STALE_WARNING_DAYS_DEFAULT:
                    st.warning(f"Data appears stale: latest date is {pd.to_datetime(last_close_date).strftime('%d-%m-%Y')} ({days_old} days old).")

            if last_close_date is not None:
                pred_date_dt = next_trading_day(last_close_date, [])
                last_close_date_str = pd.to_datetime(last_close_date).strftime("%d-%m-%Y")
                pred_date_str = pred_date_dt.strftime("%d-%m-%Y")
            else:
                last_close_date_str = "Unknown"
                pred_date_str = "Next trading day"

            pct_change = (pred / last_close - 1) * 100 if last_close != 0 else 0.0

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Last Close", f"{last_close:.2f}")
            mcol1.caption(f"Date: {last_close_date_str}    (Source column: Close)")
            mcol2.metric("Predicted Next Close", f"{pred:.2f}", delta=f"{pct_change:.2f}%")
            mcol2.caption(f"Predicted date: {pred_date_str} (next trading day)")
            mcol3.metric("Test MAE (price units)", f"{mae:.4f}")

            with st.expander("Show raw CSV last rows (verification)"):
                raw_last_rows = result.get("raw_last_rows", [])
                if raw_last_rows:
                    st.dataframe(pd.DataFrame(raw_last_rows))
                else:
                    st.write("Raw data unavailable for verification.")

            with st.expander("Raw row used for 'Last Close' (explicit)"):
                raw_last_row = result.get("raw_last_row", {})
                if raw_last_row:
                    st.write("Source column used for 'Last Close': **Close**")
                    st.dataframe(pd.DataFrame([raw_last_row]))
                else:
                    st.write("No raw_last_row available.")

            test_dates = result.get("test_dates")
            true_prices = result.get("true_test_prices")
            pred_prices = result.get("pred_test_prices")

            if test_dates is not None and len(test_dates) > 0:
                plt.figure(figsize=(12, 5))
                plt.plot(test_dates, true_prices, label="True Values", linewidth=1.5)
                plt.plot(test_dates, pred_prices, label="Predictions", linewidth=1.5)
                plt.title(f"LSTM/CNN Predictions vs True — {result['ticker']}", fontsize=14)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Close Price", fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(alpha=0.2)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()

                test_df = pd.DataFrame({"date": pd.to_datetime(test_dates).astype(str), "true_price": true_prices, "pred_price": pred_prices})
                st.download_button("Download test-series CSV", data=test_df.to_csv(index=False), file_name=f"{result['ticker']}_test_series.csv", mime="text/csv")
            else:
                st.markdown(f"### Recent Price (last {display_n} rows)")
                st.line_chart(df.set_index("date")["close"].tail(display_n))

            with st.expander("Show Processed Features (last 10 rows)"):
                st.dataframe(df.tail(10))
                st.download_button("Download processed features CSV", data=df.to_csv(index=False), file_name=f"{result['ticker']}_features.csv", mime="text/csv")

            st.success(f"✅ Predicted next close: {pred:.2f} ({pct_change:.2f}% vs last close).")

# ---------- Comparison ----------
if section == "Comparison":
    st.markdown("## ⚖️ Stock Comparison")
    left, right = st.columns(2)
    with left:
        source_l = st.selectbox("Left data source", ["Local CSV (data/)", "Upload CSV", "Yahoo Finance"], index=0, key="cmp_source_l")
        upload1 = None
        t1 = ""
        if source_l == "Local CSV (data/)":
            avail = list_data_folder_tickers()
            if avail:
                t1 = st.selectbox("Ticker A (from data/)", options=avail, index=0, key="cmp_t1")
            else:
                t1 = st.text_input("Ticker A (place CSV in data/)", value=DEFAULT_QUICK_TICKER, key="cmp_t1")
        elif source_l == "Upload CSV":
            upload1 = st.file_uploader("Upload CSV for A (left)", type=["csv"], key="cmp_upload1")
        else:
            t1 = st.text_input("Yahoo ticker A (e.g., AAPL, RELIANCE.NS)", value=DEFAULT_QUICK_TICKER, key="cmp_t1")
    with right:
        source_r = st.selectbox("Right data source", ["Local CSV (data/)", "Upload CSV", "Yahoo Finance"], index=0, key="cmp_source_r")
        upload2 = None
        t2 = ""
        if source_r == "Local CSV (data/)":
            avail2 = list_data_folder_tickers()
            if avail2:
                t2 = st.selectbox("Ticker B (from data/)", options=avail2, index=(1 if len(avail2) > 1 else 0), key="cmp_t2")
            else:
                t2 = st.text_input("Ticker B (place CSV in data/)", value="AAPL", key="cmp_t2")
        elif source_r == "Upload CSV":
            upload2 = st.file_uploader("Upload CSV for B (right)", type=["csv"], key="cmp_upload2")
        else:
            t2 = st.text_input("Yahoo ticker B (e.g., MSFT)", value="MSFT", key="cmp_t2")

    period = st.selectbox("Select Period", ["1y", "2y", "5y"], index=0, key="cmp_period")
    cmp_btn = st.button("🔍 Compare & Recommend", key="compare_button")
    if cmp_btn:
        raw1 = None
        raw2 = None
        s1 = "local" if source_l == "Local CSV (data/)" else ("upload" if source_l == "Upload CSV" else "yahoo")
        s2 = "local" if source_r == "Local CSV (data/)" else ("upload" if source_r == "Upload CSV" else "yahoo")

        if s1 == "upload":
            if upload1 is None:
                st.error("Upload CSV for left (A) or choose Local/Yahoo.")
                st.stop()
            try:
                raw1 = load_csv_from_upload(upload1)
            except Exception as e:
                st.error(f"Failed to read upload for A: {e}")
                st.stop()
        if s2 == "upload":
            if upload2 is None:
                st.error("Upload CSV for right (B) or choose Local/Yahoo.")
                st.stop()
            try:
                raw2 = load_csv_from_upload(upload2)
            except Exception as e:
                st.error(f"Failed to read upload for B: {e}")
                st.stop()

        try:
            r1 = run_pipeline_for_ticker(t1.strip().upper(), period, raw_df=raw1, dup_strategy="last", source=s1)
            r2 = run_pipeline_for_ticker(t2.strip().upper(), period, raw_df=raw2, dup_strategy="last", source=s2)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            st.stop()

        if r1.get("error") or r2.get("error"):
            st.error("One or both tickers failed. Details below:")
            st.write("Left result:", r1)
            st.write("Right result:", r2)
        else:
            df1 = r1["df"].rename(columns={"close": t1})[["date", t1]]
            df2 = r2["df"].rename(columns={"close": t2})[["date", t2]]
            merged = pd.merge(df1, df2, on="date", how="outer").sort_values("date").set_index("date")

            display_n = rows_for_period(period)
            st.markdown(f"### Price Comparison (last {display_n} points)")
            st.line_chart(merged.tail(display_n))

            score1 = (r1["pred_next_close"] / r1["last_close"] - 1) / max(r1["mae"], 1e-9)
            score2 = (r2["pred_next_close"] / r2["last_close"] - 1) / max(r2["mae"], 1e-9)
            best = t1 if score1 > score2 else t2

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{t1}**")
                st.metric("Last Close", f"{r1['last_close']:.2f}")
                lc_date1 = pd.to_datetime(r1.get("last_close_date")).strftime("%d-%m-%Y") if r1.get("last_close_date") is not None else "Unknown"
                pred_date1 = next_trading_day(r1.get("last_close_date"), []).strftime("%d-%m-%Y") if r1.get("last_close_date") is not None else "Next trading day"
                st.caption(f"Date: {lc_date1}    (Source: {r1.get('source')})")
                st.metric("Predicted Next Close", f"{r1['pred_next_close']:.2f}")
                st.caption(f"Predicted date: {pred_date1} (next trading day)")
                st.metric("Test MAE", f"{r1['mae']:.4f}")
            with col2:
                st.markdown(f"**{t2}**")
                st.metric("Last Close", f"{r2['last_close']:.2f}")
                lc_date2 = pd.to_datetime(r2.get("last_close_date")).strftime("%d-%m-%Y") if r2.get("last_close_date") is not None else "Unknown"
                pred_date2 = next_trading_day(r2.get("last_close_date"), []).strftime("%d-%m-%Y") if r2.get("last_close_date") is not None else "Next trading day"
                st.caption(f"Date: {lc_date2}    (Source: {r2.get('source')})")
                st.metric("Predicted Next Close", f"{r2['pred_next_close']:.2f}")
                st.caption(f"Predicted date: {pred_date2} (next trading day)")
                st.metric("Test MAE", f"{r2['mae']:.4f}")

            merged_download_df = merged.tail(display_n).reset_index()
            st.download_button("Download comparison CSV (displayed)", merged_download_df.to_csv(index=False), file_name=f"comparison_{t1}_{t2}.csv", mime="text/csv")

            st.success(f"📊 Recommendation: **{best}** seems better based on predicted return vs. error.")

# ---------- Bulk Download (Excel export) ----------
if section == "Bulk Download":
    st.markdown("## 📥 Bulk: Predict & Download Tomorrow's Close for Multiple Companies (Excel)")
    st.write("Select companies from `data/`, upload files, or add Yahoo tickers (comma-separated). The app will run the pipeline for each selected item and produce an Excel table.")

    col1, col2 = st.columns([2, 1])
    with col1:
        available = list_data_folder_tickers()
        multi_select = st.multiselect("Select tickers from data/ (multi-select allowed):", options=available, default=available[:5])
        uploaded_files = st.file_uploader("Or upload multiple CSV files (optional)", type=["csv"], accept_multiple_files=True)
        yahoo_text = st.text_area("Or enter Yahoo tickers (comma-separated), e.g. AAPL, MSFT, RELIANCE.NS", value="", height=80)
    with col2:
        preview_rows = st.number_input("Preview rows in verification (per ticker)", min_value=1, max_value=20, value=3)
        run_bulk = st.button("🔁 Run predictions & build Excel package")

    if run_bulk:
        tasks: List[Dict[str, Any]] = []
        for t in multi_select:
            tasks.append({"name": t, "raw_df": None, "local": True, "source": "local"})
        if uploaded_files:
            for f in uploaded_files:
                try:
                    raw_df = pd.read_csv(f)
                except Exception as e:
                    st.error(f"Failed to read uploaded file {getattr(f, 'name', 'uploaded')} : {e}")
                    raw_df = None
                name = os.path.splitext(getattr(f, "name", "UPLOAD"))[0]
                tasks.append({"name": name, "raw_df": raw_df, "local": False, "source": "upload"})
        yahoo_list = [s.strip().upper() for s in yahoo_text.replace("\n", ",").split(",") if s.strip() != ""]
        for y in yahoo_list:
            tasks.append({"name": y, "raw_df": None, "local": False, "source": "yahoo"})

        if not tasks:
            st.warning("No tickers selected, files uploaded, or yahoo tickers provided.")
            st.stop()

        results_rows = []
        progress = st.progress(0)
        total = len(tasks)
        i = 0
        for tt in tasks:
            i += 1
            progress.progress(int((i-1)/total*100))
            name = tt["name"]
            rawdf = tt["raw_df"]
            src = tt.get("source", "local")
            st.info(f"Processing {name} ({i}/{total}) [source: {src}] ...")
            try:
                if src == "yahoo":
                    res = run_pipeline_for_ticker(ticker=name, period="1y", raw_df=None, dup_strategy="last", source="yahoo")
                elif src == "upload":
                    res = run_pipeline_for_ticker(ticker="", period="1y", raw_df=rawdf, dup_strategy="last", source="upload")
                else:
                    res = run_pipeline_for_ticker(ticker=name, period="1y", raw_df=None, dup_strategy="last", source="local")
            except Exception as e:
                res = {"error": "exception", "message": str(e)}

            if res.get("error"):
                status = f"error: {res.get('error')}"
                last_close = None
                last_date = None
                pred_close = None
                pred_date = None
                mae = None
            else:
                last_close = res.get("last_close")
                last_date = res.get("last_close_date")
                pred_close = res.get("pred_next_close")
                try:
                    pred_dt = next_trading_day(last_date, []) if last_date is not None else None
                    pred_date = pd.to_datetime(pred_dt).strftime("%Y-%m-%d") if pred_dt is not None else None
                except Exception:
                    pred_date = None
                mae = res.get("mae")
                status = "ok"

            results_rows.append({
                "company": name,
                "predicted_date": pred_date,
                "predicted_close": pred_close,
                "last_trading_date": pd.to_datetime(last_date).strftime("%Y-%m-%d") if last_date is not None else None,
                "last_close": last_close,
                "mae": mae,
                "status": status
            })
            st.success(f"Finished {name}: {status}")
        progress.progress(100)

        results_df = pd.DataFrame(results_rows)
        st.markdown("### Results table (preview)")
        st.dataframe(results_df)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_bytes, file_name="bulk_predicted_closes.csv", mime="text/csv")

        to_write = io.BytesIO()
        with pd.ExcelWriter(to_write, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="predictions", index=False)
            meta = pd.DataFrame({
                "generated_at": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                "dup_strategy": ["last"],
                "holiday_count": [0]
            })
            meta.to_excel(writer, sheet_name="meta", index=False)
            writer.save()
        to_write.seek(0)
        st.download_button("Download results Excel (.xlsx)", data=to_write.getvalue(), file_name="bulk_predicted_closes.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("Bulk processing complete — Excel and CSV downloads are ready.")

# ---------------------------
# Settings & Important Notes (moved to bottom)
# ---------------------------
st.markdown("---")
with st.expander("Settings & Important Notes (click to open)"):
    st.markdown("### Settings (moved from sidebar)")
    st.write("- Duplicate-date resolution uses 'last' by default when CSV contains multiple rows per calendar day.")
    st.write("- Holiday calendar: put `holidays.csv` in `./data/` or upload; used to compute next trading-day.")
    st.write("- Data sources: Upload CSV, Local CSV (`./data/<TICKER>.csv`), Yahoo Finance (internet required).")
    st.write("- Bulk Download creates Excel with `predictions` sheet (company, predicted_date, predicted_close, last_trading_date, last_close, mae, status).")
    st.markdown("### Debug note for Yahoo failures")
    st.write("If you see 'No data from Yahoo for <TICKER>' the diagnostic message now includes the strategies attempted and any exceptions encountered. Common fixes:")
    st.write("1. Check internet connection.")
    st.write("2. Try adding the exchange suffix (e.g., `TCS.NS`).")
    st.write("3. If you still see failures, download CSV manually and use Upload CSV or Local CSV mode.")
st.caption("Sidebar: only quick picks; all other settings are below.")
