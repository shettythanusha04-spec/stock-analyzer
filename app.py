import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Analyzer", layout="wide")

COMPANIES = {
    "Glenmark Pharmaceuticals": "GLENMARK.NS",
    "Phoenix Mills": "PHOENIXLTD.NS",
    "Fortis Healthcare": "FORTIS.NS",
    "Alkem Laboratories": "ALKEM.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
}


@st.cache_data(ttl=3600)
def load_prices(ticker):
    df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None)
    return df


def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def period_return(close, days):
    if len(close) <= days:
        return np.nan
    return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


# ---------------- UI ----------------
st.title("📈 Stock Analyzer: 5 Selected Companies")
st.caption("Technical indicators, momentum, and performance vs Nifty 50. Educational use only, not investment advice.")

company = st.sidebar.selectbox("Select company", list(COMPANIES.keys()))
ticker = COMPANIES[company]
st.subheader(f"{company} ({ticker})")
st.sidebar.markdown("**Features**\n- RSI\n- MACD + crossover\n- Bollinger Bands\n- Volume analysis\n- Alpha vs Nifty\n- 52-week high distance\n- Monthly returns heatmap")

df = load_prices(ticker)
if df.empty:
    st.error("No data found for this ticker. Try another one.")
    st.stop()

close, vol = df["Close"], df["Volume"]
nifty = load_prices("^NSEI")["Close"]

# Summary metrics
rsi_s = rsi(close)
macd_line, macd_sig, macd_hist = macd(close)
mid, up, lo = bollinger(close)
high52 = close.tail(252).max()
pct_high = close.iloc[-1] / high52 * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", f"₹{close.iloc[-1]:,.2f}", f"{close.pct_change().iloc[-1]*100:.2f}%")
c2.metric("RSI (14)", f"{rsi_s.iloc[-1]:.1f}")
c3.metric("% of 52-wk high", f"{pct_high:.1f}%")
c4.metric("Trading days", len(close))

tabs = st.tabs(["Price & Bollinger", "RSI", "MACD", "Volume", "vs Nifty", "Monthly heatmap"])

with tabs[0]:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(close.index, close, label="Close", color="black")
    ax.plot(mid.index, mid, label="20-day mean", color="blue", lw=1)
    ax.fill_between(close.index, lo, up, color="grey", alpha=0.2, label="Bollinger band")
    ax.legend()
    st.pyplot(fig)
    pb = (close.iloc[-1] - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1])
    if pb > 0.8:
        st.info(f"%B = {pb:.2f}: near upper band (strong / overbought zone)")
    elif pb < 0.2:
        st.info(f"%B = {pb:.2f}: near lower band (weak / oversold zone)")
    else:
        st.info(f"%B = {pb:.2f}: inside the bands (neutral)")

with tabs[1]:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rsi_s.index, rsi_s, color="purple")
    ax.axhline(70, color="red", ls="--")
    ax.axhline(30, color="green", ls="--")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    r = rsi_s.iloc[-1]
    label = "Overbought" if r > 70 else "Oversold" if r < 30 else "Bullish momentum" if r > 50 else "Weak / bearish"
    st.info(f"RSI = {r:.1f}: {label}")

with tabs[2]:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(macd_line.index, macd_line, label="MACD")
    ax.plot(macd_sig.index, macd_sig, label="Signal")
    ax.bar(macd_hist.index, macd_hist, color=np.where(macd_hist >= 0, "green", "red"), alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    diff = (macd_line - macd_sig).tail(10)
    if diff.iloc[0] < 0 and diff.iloc[-1] > 0:
        st.success("Bullish crossover in the last 10 days")
    elif diff.iloc[0] > 0 and diff.iloc[-1] < 0:
        st.error("Bearish crossover in the last 10 days")
    else:
        st.info("Momentum accelerating up" if macd_hist.iloc[-1] > 0 else "Momentum accelerating down")

with tabs[3]:
    recent = vol.tail(20)
    ratio = recent.iloc[-1] / vol.tail(60).mean()
    rets = close.pct_change().tail(20)
    upv = recent[rets > 0].mean()
    dnv = recent[rets < 0].mean()
    acc = upv / dnv if dnv and not np.isnan(dnv) else np.nan
    a, b = st.columns(2)
    a.metric("Volume vs 60-day avg", f"{ratio:.2f}x")
    b.metric("Accumulation ratio (up-day / down-day vol)", f"{acc:.2f}" if not np.isnan(acc) else "n/a")
    st.bar_chart(vol.tail(60))

with tabs[4]:
    rows = []
    for name, d in [("1 month", 21), ("3 months", 63), ("1 year", 252)]:
        s, n = period_return(close, d), period_return(nifty, d)
        rows.append({"Period": name, "Stock %": round(s, 2), "Nifty %": round(n, 2), "Alpha %": round(s - n, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    both = pd.concat([close, nifty], axis=1, keys=["Stock", "Nifty"]).dropna()
    st.line_chart(both / both.iloc[0] * 100)
    st.caption("Both rebased to 100 at the start.")

with tabs[5]:
    m = close.resample("ME").last().pct_change().dropna() * 100
    table = pd.DataFrame({"Year": m.index.year, "Month": m.index.strftime("%b"), "Ret": m.values})
    pivot = table.pivot(index="Year", columns="Month", values="Ret")
    order = [x for x in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] if x in pivot.columns]
    st.dataframe(pivot[order].style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}", na_rep="-"))
