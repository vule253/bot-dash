import logging
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import subprocess
import sys
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if "bot_process" not in st.session_state:
    st.session_state["bot_process"] = None

# Lấy danh sách top 100 coin từ CoinGecko và lọc stablecoin
@st.cache_data(ttl=3600)
def get_top_coins_from_coingecko(limit=100):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"CoinGecko API error: {response.status_code} - {response.text}")
        return []
    data = response.json()
    if not isinstance(data, list):
        st.error(f"Dữ liệu trả về không đúng định dạng: {data}")
        return []
    stablecoins = {"USDT", "USDC", "BUSD", "DAI", "UST", "TUSD", "EURS"}
    coin_symbols = []
    for coin in data:
        if not isinstance(coin, dict) or "symbol" not in coin:
            continue
        sym = coin["symbol"].upper()
        if sym not in stablecoins:
            coin_symbols.append(sym)
    return coin_symbols

all_top_coins = get_top_coins_from_coingecko(limit=100)
#if not all_top_coins:
#    all_top_coins = ["BTC", "ETH", "ADA", "XRP", "BNB", "SOL"]

# Danh sách coin được chọn; nếu cần, bạn có thể thêm lọc theo sàn KuCoin hỗ trợ
coin_options = all_top_coins  
selected_coins = st.sidebar.multiselect(
    "Chọn các đồng coin để trading (tối đa 3):",
    options=coin_options,
    default=["BTC", "ETH", "ADA"]
)
if len(selected_coins) > 3:
    st.sidebar.error("Chỉ được chọn tối đa 3 coin.")

# Hiển thị mô hình của coin đầu tiên được chọn (hoặc hiển thị log riêng cho từng coin)
default_coin = selected_coins[0] if selected_coins else "BTC"
model_file = f"trading_signal_model_{default_coin}.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)
    model_status = f"✅ {default_coin}: Model Loaded Successfully"
else:
    model_status = f"❌ {default_coin}: No Trained Model Found"

st.title("📊 Crypto Trading Bot Dashboard")
st.markdown("### ⚡ Model Training Status")
st.text(model_status)

st.sidebar.markdown("### ⚙️ Trading Settings")
amount = st.sidebar.number_input("Trade Amount (USDT)", min_value=1.0, max_value=1000.0, value=10.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 3) / 100
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1, 10, 2) / 100
take_profit_pct = st.sidebar.slider("Take Profit (%)", 1, 10, 4) / 100

bot_status = st.sidebar.empty()
train_status = st.sidebar.empty()

def start_bot():
    if st.session_state["bot_process"] is None:
        env = os.environ.copy()
        env["TRADE_AMOUNT"] = str(amount)
        env["RISK_PER_TRADE"] = str(risk_per_trade)
        env["STOP_LOSS_PCT"] = str(stop_loss_pct)
        env["TAKE_PROFIT_PCT"] = str(take_profit_pct)
        coin_param = ",".join(selected_coins)
        st.session_state["bot_process"] = subprocess.Popen(
            ["python", "app_bot.py", "--coins", coin_param],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        bot_status.success("🚀 Bot is running!")
    else:
        bot_status.warning("⚠️ Bot is already running!")

def stop_bot():
    if st.session_state["bot_process"] is not None:
        st.session_state["bot_process"].terminate()
        st.session_state["bot_process"].wait()
        st.session_state["bot_process"] = None
        bot_status.success("⛔ Bot stopped!")
    else:
        bot_status.warning("⚠️ No bot is currently running!")

def train_model():
    train_status.info("🔄 Training model for selected coin(s)...")
    responses = {}
    for coin in selected_coins:
        process = subprocess.run(
        [sys.executable, "daily_train.py", "--coin", coin, "--once"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
        )
        responses[coin] = process.stdout if process.returncode == 0 else process.stderr
    output_text = "\n\n".join([f"{coin}:\n{res}" for coin, res in responses.items()])
    train_status.success("✅ Model training completed!")
    st.text_area("Training Log", output_text, height=300)

st.sidebar.markdown("### 🎮 Bot Control")
if st.sidebar.button("▶ Start Bot"):
    start_bot()
if st.sidebar.button("⏹ Stop Bot"):
    stop_bot()
if st.sidebar.button("🔄 Train Model"):
    train_model()

if st.session_state["bot_process"] is None:
    bot_status.warning("❌ Bot is not running.")
else:
    bot_status.success("✅ Bot is running!")

def load_train_log_for_coin(coin):
    log_file = f"train_log_{coin}.txt"
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file, sep="|", names=["Model", "F1-Score", "Timestamp"], parse_dates=["Timestamp"])
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error reading train log for {coin}: {e}")
            return pd.DataFrame(columns=["Model", "F1-Score", "Timestamp"])
    return pd.DataFrame(columns=["Model", "F1-Score", "Timestamp"])

for coin in selected_coins:
    log_df = load_train_log_for_coin(coin)
    st.write(f"### 📈 {coin} Training Performance")
    if not log_df.empty:
        st.dataframe(log_df.tail(5))
        fig, ax = plt.subplots()
        ax.plot(log_df["Timestamp"], log_df["F1-Score"], marker='o', linestyle='-', color='b')
        ax.set_xlabel("Training Date")
        ax.set_ylabel("F1-Score")
        ax.set_title(f"{coin} Model Performance Over Time")
        st.pyplot(fig)
    else:
        st.warning(f"No valid training logs found for {coin}!")

def load_profit_log():
    if os.path.exists("profit_log.txt"):
        return pd.read_csv("profit_log.txt", sep="|", names=["Symbol", "Side", "Entry", "Exit", "Profit", "Time"], parse_dates=["Time"])
    return pd.DataFrame(columns=["Symbol", "Side", "Entry", "Exit", "Profit", "Time"])

profit_log = load_profit_log()
if not profit_log.empty:
    st.write("### 💰 Trading Profit/Loss")
    total_profit = profit_log["Profit"].sum()
    win_rate = (profit_log[profit_log["Profit"] > 0].shape[0] / profit_log.shape[0]) * 100
    losing_trades = profit_log[profit_log["Profit"] < 0]
    if not losing_trades.empty and losing_trades["Profit"].mean() != 0:
        risk_reward_ratio = abs(profit_log["Profit"].mean() / losing_trades["Profit"].mean())
    else:
        risk_reward_ratio = float("nan")
    st.metric(label="Total Profit", value=f"${total_profit:.2f}")
    st.metric(label="Win Rate", value=f"{win_rate:.2f}%")
    st.metric(label="Risk/Reward Ratio", value=f"{risk_reward_ratio:.2f}")
    st.dataframe(profit_log.tail(10))
    
    fig, ax = plt.subplots()
    ax.plot(profit_log["Time"], profit_log["Profit"].cumsum(), marker='o', linestyle='-', color='green')
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("Profit Over Time")
    st.pyplot(fig)
    
    st.write("### 📊 Trade Entries & Exits")
    fig, ax = plt.subplots()
    ax.scatter(profit_log["Time"], profit_log["Entry"], label="Entry Price", color='blue', marker='o')
    ax.scatter(profit_log["Time"], profit_log["Exit"], label="Exit Price", color='red', marker='x')
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("No profit logs found!")

st.write("### 🟢 Live Signal Prediction")
st.info("To display real-time signals, integrate this with the bot's live trading script.")

st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.write("Adjust bot settings here in future versions!")
