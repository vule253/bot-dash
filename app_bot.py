import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import logging
import threading
import joblib
#import talib/ Bỏ talib dùng ta
from ta.volatility import AverageTrueRange
import argparse
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator  # ADXIndicator đã chuyển sang ta.trend
from retry import retry

# Sử dụng st.secrets để lấy thông tin API (các giá trị này sẽ được cấu hình qua secrets.toml trên Streamlit Cloud)
api_key = st.secrets["KUCOIN_API_KEY"]
secret = st.secrets["KUCOIN_SECRET"]
password = st.secrets["KUCOIN_PASSPHRASE"]

# Khởi tạo kết nối KuCoin
kucoin = ccxt.kucoin({
    "apiKey": api_key,
    "secret": secret,
    "password": password,
    "enableRateLimit": True,
})

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(tries=3, delay=2)
def get_klines(symbol, timeframe="1m", limit=50):
    try:
        ohlcv = kucoin.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"Error while fetching OHLCV data for {symbol}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    try:
        df["EMA9"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["EMA21"] = EMAIndicator(df["close"], window=21).ema_indicator()
        df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
        df["MACD"] = MACD(df["close"]).macd()
        df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["ATR"] = atr_indicator.average_true_range()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error while adding indicators: {e}")
        return df

# Tải mô hình và label encoder riêng cho từng coin
def load_model_and_encoder(coin):
    try:
        model = joblib.load(f"trading_signal_model_{coin}.pkl")
        le = joblib.load(f"label_encoder_{coin}.pkl")
        return model, le
    except Exception as e:
        logging.error(f"Error loading model/encoder for {coin}: {e}")
        return None, None

def predict_signal_for_coin(coin, df):
    model, le = load_model_and_encoder(coin)
    if model is None or le is None:
        return "HOLD"
    try:
        feature_cols = ["EMA9", "EMA21", "RSI", "MACD", "ADX"]
        X_new = df[feature_cols].values[-1].reshape(1, -1)
        signal_numeric = model.predict(X_new)[0]
        signal = le.inverse_transform([signal_numeric])[0]
        return signal
    except Exception as e:
        logging.error(f"Error predicting signal for {coin}: {e}")
        return "HOLD"

def calculate_sl_tp(entry_price, side, atr):
    try:
        sl = entry_price - 1.5 * atr if side == "BUY" else entry_price + 1.5 * atr
        tp = entry_price + 2.5 * atr if side == "BUY" else entry_price - 2.5 * atr
        return round(sl, 2), round(tp, 2)
    except Exception as e:
        logging.error(f"Error calculating SL/TP for {entry_price}: {e}")
        return entry_price, entry_price

def get_order_book(symbol):
    try:
        order_book = kucoin.fetch_order_book(symbol)
        bid_volume = sum([x[1] for x in order_book["bids"]])
        ask_volume = sum([x[1] for x in order_book["asks"]])
        return bid_volume, ask_volume
    except Exception as e:
        logging.error(f"Error while fetching order book for {symbol}: {e}")
        return 0, 0

def kelly_criterion(win_rate, risk_reward_ratio):
    try:
        return max(0, (win_rate - (1 - win_rate) / risk_reward_ratio))
    except Exception as e:
        logging.error(f"Error calculating Kelly Criterion: {e}")
        return 0

# Hàm lấy số dư tài khoản thực (USDT)
def get_account_balance(currency="USDT"):
    try:
        balance = kucoin.fetch_balance()
        return balance["free"].get(currency, 0)
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        return 0

# Hàm chạy bot cho một coin cụ thể
def run_bot_for_coin(coin, allocation=0.33):
    symbol = f"{coin}/USDT"
    logging.info(f"Starting bot for {coin}...")
    
    # Lấy số dư thực tế và phân bổ vốn cho coin này
    total_balance = get_account_balance("USDT")
    allocated_balance = total_balance * allocation
    win_rate, risk_reward_ratio = 0.6, 2
    trade_size = allocated_balance * kelly_criterion(win_rate, risk_reward_ratio)
    
    while True:
        try:
            df = get_klines(symbol)
            df = add_indicators(df)
            if df.empty:
                logging.warning(f"No data for {symbol}")
                time.sleep(60)
                continue
            signal = predict_signal_for_coin(coin, df)
            bid_vol, ask_vol = get_order_book(symbol)
            if signal in ["BUY", "SELL"] and ask_vol > 0 and (bid_vol / ask_vol > 1.2):
                place_order(symbol, signal, trade_size, df["ATR"].iloc[-1])
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error in bot for {coin}: {e}")
            time.sleep(60)

# Hàm chạy bot cho nhiều coin cùng lúc sử dụng threading
def run_bot_multi(coins, allocation_per_coin=0.33):
    threads = []
    for coin in coins:
        t = threading.Thread(target=run_bot_for_coin, args=(coin, allocation_per_coin))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def place_order(symbol, side, quantity, atr):
    try:
        ticker = kucoin.fetch_ticker(symbol)
        entry_price = ticker["last"]
        sl_price, tp_price = calculate_sl_tp(entry_price, side, atr)
        kucoin.create_order(symbol, "market", side.lower(), quantity)
        logging.info(f"Placed {side} order for {symbol} at {entry_price}")
        threading.Thread(target=update_trailing_stop, args=(symbol, side, sl_price, 0.01, quantity)).start()
    except Exception as e:
        logging.error(f"Error while placing order for {symbol}: {e}")

def update_trailing_stop(symbol, side, current_sl, trailing_pct, quantity):
    try:
        while True:
            ticker = kucoin.fetch_ticker(symbol)
            current_price = ticker["last"]
            if side == "BUY" and current_price * (1 - trailing_pct) > current_sl:
                current_sl = current_price * (1 - trailing_pct)
                kucoin.create_order(symbol, "stop_market", "sell", quantity, params={"stopPrice": current_sl})
            elif side == "SELL" and current_price * (1 + trailing_pct) < current_sl:
                current_sl = current_price * (1 + trailing_pct)
                kucoin.create_order(symbol, "stop_market", "buy", quantity, params={"stopPrice": current_sl})
            time.sleep(10)
    except Exception as e:
        logging.error(f"Error while updating trailing stop for {symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading bot for multiple coins")
    parser.add_argument("--coins", type=str, default="BTC,ETH,ADA", help="Comma separated list of coin symbols (e.g., BTC,ETH,ADA)")
    args = parser.parse_args()
    coins = [coin.strip() for coin in args.coins.split(",")]
    run_bot_multi(coins)
