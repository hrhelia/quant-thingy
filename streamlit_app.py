import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

st.set_page_config(page_title="Quant Finance AI", layout="wide")
st.title("📈 Belfort AI - Ensemble Trading Intelligence")

# ===== Helper Functions =====

def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the given symbol and date range.")
        st.stop()
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def add_indicators(df):
    df = df.copy()
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily Return"].rolling(window=30).std()
    df["Resistance"] = df["High"].rolling(window=7).max()
    df["Support"] = df["Low"].rolling(window=30).min()
    return df

def make_decision(data):
    if len(data) < 30:
        return "HOLD", None, None, None, None, []
    
    latest = data.iloc[-1]
    conditions = []
    try:
        sma_7 = float(latest["SMA_7"])
        sma_30 = float(latest["SMA_30"])
        vol = float(latest["Volatility"])
        close = float(latest["Close"])
        support = float(latest["Support"])
    except Exception:
        return "HOLD", None, None, None, None, []
    
    conditions.append("✅ SMA_7 > SMA_30" if sma_7 > sma_30 else "❌ SMA_7 <= SMA_30")
    conditions.append("✅ Volatility < 2%" if vol < 0.02 else "❌ Volatility >= 2%")
    conditions.append("✅ Close > Support" if close > support else "❌ Close <= Support")
    
    if all(c.startswith("✅") for c in conditions):
        sl = support
        rr_2_tp = close + 2 * (close - sl)
        rr_3_tp = close + 3 * (close - sl)
        if (rr_3_tp - close) > (rr_2_tp - close):
            return "BUY", close, rr_3_tp, sl, 3, conditions
        else:
            return "BUY", close, rr_2_tp, sl, 2, conditions
    else:
        sl = close * 0.98
        rr = 2
        tp = close + rr * (close - sl)
        return "HOLD", close, tp, sl, rr, conditions

def simple_backtest(df, decision_func):
    trades = []
    equity = [100]
    equity_dates = []

    df_bt = df.copy().dropna().reset_index()

    for i in range(30, len(df_bt) - 1):
        sub_df = df_bt.iloc[:i+1]
        decision, entry, tp, sl, rr, conditions = decision_func(sub_df)

        if decision == "BUY":
            entry_date = df_bt.iloc[i]["Date"]
            try:
                entry = float(entry)
                tp = float(tp)
                sl = float(sl)
            except Exception:
                continue

            for j in range(i+1, len(df_bt)):
                price = float(df_bt.iloc[j]["Close"])
                if price >= tp:
                    exit_price = tp
                    result = "TP"
                    break
                elif price <= sl:
                    exit_price = sl
                    result = "SL"
                    break
            else:
                continue

            exit_date = df_bt.iloc[j]["Date"]
            gain_pct = ((exit_price - entry) / entry) * 100
            equity.append(equity[-1] * (1 + gain_pct / 100.0))
            equity_dates.append(exit_date)

            trades.append({
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Entry Price": entry,
                "Exit Price": exit_price,
                "Result": result,
                "RR": rr,
                "Gain %": gain_pct
            })

    trades_df = pd.DataFrame(trades)

    if len(equity[1:]) != len(equity_dates):
        equity_curve = pd.Series(dtype='float64')
    else:
        equity_curve = pd.Series(equity[1:], index=equity_dates)

    return trades_df, equity_curve

def forecast_prices_xgb(df):
    df_model = df.dropna().copy()

    df_model["Close_t-1"] = df_model["Close"].shift(1)
    df_model["Close_t-2"] = df_model["Close"].shift(2)
    df_model.dropna(inplace=True)

    features = ["Close_t-1", "Close_t-2", "SMA_7", "SMA_30", "Volatility", "Support", "Resistance"]
    X = df_model[features]
    y = df_model["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    y_pred_full = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred_full))
    mae = mean_absolute_error(y, y_pred_full)
    mape = np.mean(np.abs((y - y_pred_full) / y)) * 100
    r2 = r2_score(y, y_pred_full)
    n, p = len(y), X.shape[1]
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    future_dates = [df_model.index.max() + datetime.timedelta(days=i) for i in range(1, 366)]
    last_row = df_model.iloc[-1:].copy()

    future_preds = []
    for _ in future_dates:
        new_features = last_row[features].values
        pred = model.predict(new_features)[0]
        future_preds.append(pred)
        last_row["Close_t-2"] = last_row["Close_t-1"]
        last_row["Close_t-1"] = pred
        last_row["Close"] = pred

    return df_model.index, y_pred_full, future_dates, future_preds, {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adj_r2": adj_r2
    }

def plot_charts(df, model_index, y_pred, future_dates, y_future, equity_curve, trades_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_7"], name="SMA 7", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_30"], name="SMA 30", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=model_index, y=y_pred, name="XGBoost Fit", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=future_dates, y=y_future, name="XGBoost Forecast", line=dict(color="purple", dash="dot")))

    if not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df["Entry Date"], y=trades_df["Entry Price"],
            mode='markers', name="BUY", marker=dict(color='green', symbol='triangle-up', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=trades_df["Exit Date"], y=trades_df["Exit Price"],
            mode='markers', name="SELL", marker=dict(color='red', symbol='x', size=10)
        ))

    fig.update_layout(title="Price, Trades & XGBoost Forecast", xaxis_title="Date", yaxis_title="Price", width=1100, height=600)
    st.plotly_chart(fig, use_container_width=True)

    if not equity_curve.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name="Equity ($)", line=dict(color="red")))
        fig_eq.update_layout(title="Simulated Account Growth", xaxis_title="Date", yaxis_title="Equity ($)", width=1100, height=500)
        st.plotly_chart(fig_eq, use_container_width=True)

# ===== Main Program =====

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if st.button("Run Analysis"):
    with st.spinner("Fetching data..."):
        data = load_data(symbol, start_date, end_date)
    data = add_indicators(data)

    decision, entry, tp, sl, rr, checks = make_decision(data)
    st.subheader("📌 Trading Decision")
    for check in checks:
        st.write(check)
    if decision == "BUY":
        st.success(f"💰 BUY @ ${entry:.2f} | TP: ${tp:.2f} | SL: ${sl:.2f} | RR: {rr}:1")
    else:
        st.warning(f"⚠️ HOLD - Not optimal to buy. RR: {rr}:1")
        if entry:
            st.write(f"Hypothetical Entry: ${entry:.2f} | TP: ${tp:.2f} | SL: ${sl:.2f}")

    st.subheader("📦 Backtesting (Paper-Trading Simulation)")
    bt_results, equity_curve = simple_backtest(data, make_decision)
    if not bt_results.empty:
        st.dataframe(bt_results)
        win_rate = (bt_results["Result"] == "TP").mean() * 100
        avg_gain = bt_results["Gain %"].mean()
        st.markdown(f"**✅ Win Rate:** {win_rate:.2f}%")
        st.markdown(f"**📈 Avg Gain per Trade:** {avg_gain:.2f}%")
    else:
        st.info("No trades triggered during the selected timeframe.")

    st.subheader("📊 Forecasting with Ensemble Model")
    model_index, y_pred_full, future_dates, y_future, forecast_metrics = forecast_prices_xgb(data)
    st.write(f"R² Score: {forecast_metrics['r2']:.4f}")
    st.write(f"Adjusted R²: {forecast_metrics['adj_r2']:.4f}")
    st.write(f"RMSE: {forecast_metrics['rmse']:.4f}")
    st.write(f"MAE: {forecast_metrics['mae']:.4f}")
    st.write(f"MAPE: {forecast_metrics['mape']:.2f}%")

    st.subheader("📉 Historical, Trades & Forecast Charts")
    plot_charts(data, model_index, y_pred_full, future_dates, y_future, equity_curve, bt_results)
