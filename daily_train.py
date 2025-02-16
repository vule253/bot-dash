import ccxt
import pandas as pd
import numpy as np
import logging
import joblib
import schedule
import time
import argparse
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from joblib import Memory

# Caching dữ liệu (cache sẽ lưu vào thư mục ./cachedir, với TTL mặc định là vô hạn)
memory = Memory(location="./cachedir", verbose=0)

# Cấu hình logging
log_path = os.path.join(os.getcwd(), "train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logging.info("Current Working Directory: %s", os.getcwd())

# Ghi log bắt đầu training
try:
    with open(log_path, "a") as f:
        f.write("Test Log: Training started\n")
except Exception as e:
    logging.error(f"Error while writing log: {e}")

#@memory.cache
def fetch_historical_data(symbol="BTC/USDT", timeframe="5m", limit=5000, since=None):
    try:
        kucoin = ccxt.kucoin({"enableRateLimit": True})
        if since is None:
            ohlcv = kucoin.fetch_ohlcv(symbol, timeframe, limit=limit)
        else:
            ohlcv = kucoin.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"Error while fetching historical data: {e}")
        return pd.DataFrame()

def add_indicators(df):
    try:
        df["EMA9"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["EMA21"] = EMAIndicator(df["close"], window=21).ema_indicator()
        df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
        df["MACD"] = MACD(df["close"]).macd()
        df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        df["OBV"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        # Sử dụng AverageTrueRange từ thư viện ta để thay thế talib.ATR
        atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["ATR"] = atr_indicator.average_true_range()
        df["BB_Upper"] = BollingerBands(df["close"]).bollinger_hband()
        df["BB_Lower"] = BollingerBands(df["close"]).bollinger_lband()
        df["WilliamsR"] = WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error while adding indicators: {e}")
        return df

def create_labels(df, threshold=0.002):
    try:
        df["future_return"] = df["close"].pct_change().shift(-1)
        conditions = [df["future_return"] > threshold, df["future_return"] < -threshold]
        choices = ["BUY", "SELL"]
        df["signal"] = np.select(conditions, choices, default="HOLD")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error while creating labels: {e}")
        return df

def prepare_features_labels(df):
    try:
        feature_cols = ["EMA9", "EMA21", "RSI", "MACD", "ADX", "OBV", "ATR", "BB_Upper", "BB_Lower", "WilliamsR"]
        X = df[feature_cols].copy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["signal"])
        return X, y, label_encoder
    except Exception as e:
        logging.error(f"Error while preparing features and labels: {e}")
        return pd.DataFrame(), np.array([]), None

def train_and_select_best_model(threshold=0.002, coin="BTC"):
    start_time = time.time()
    try:
        df = fetch_historical_data(symbol=f"{coin}/USDT", limit=5000)
        if df.empty:
            logging.error("No data fetched. Skipping model training.")
            return
        
        df = add_indicators(df)
        df = create_labels(df, threshold=threshold)
        X, y, label_encoder = prepare_features_labels(df)
        if X.empty or len(np.unique(y)) < 2:
            logging.error("Not enough data or only one class found. Skipping training.")
            return

        logging.info(f"Class distribution before SMOTE: {pd.Series(y).value_counts()}")
        class_counts = pd.Series(y).value_counts()
        min_count = class_counts.min()
        if min_count < 2:
            logging.warning("Min class has <2 samples. Skipping SMOTE entirely.")
            X_res, y_res = X, y
        elif min_count == 2:
            smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=1)
            X_res, y_res = smote.fit_resample(X, y)
        else:
            smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=2)
            X_res, y_res = smote.fit_resample(X, y)

        # Lưu label encoder cho coin
        joblib.dump(label_encoder, f"label_encoder_{coin}.pkl")
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Xây dựng các Pipeline với StandardScaler
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        pipeline_xgb = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])
        pipeline_lgb = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', lgb.LGBMClassifier(random_state=42))
        ])
        pipeline_mlp = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(max_iter=800, random_state=42))
        ])

        models_params = {
            "RandomForest": (pipeline_rf, {
                "classifier__n_estimators": [100, 300, 500],
                "classifier__max_depth": [None, 20, 50, 80]
            }),
            "XGBoost": (pipeline_xgb, {
                "classifier__n_estimators": [100, 300, 500],
                "classifier__max_depth": [3, 6, 12, 15],
                "classifier__learning_rate": [0.01, 0.1]
            }),
            "LightGBM": (pipeline_lgb, {
                "classifier__n_estimators": [100, 300, 500],
                "classifier__num_leaves": [31, 70, 120],
                "classifier__learning_rate": [0.01, 0.1]
            }),
            "MLP": (pipeline_mlp, {
                "classifier__hidden_layer_sizes": [(50, 50), (100,), (100, 100)],
                "classifier__learning_rate_init": [0.001, 0.01, 0.0001]
            })
        }

        best_model, best_score, best_model_name = None, -1, ""
        for name, (model, param_grid) in models_params.items():
            search = RandomizedSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, n_iter=10, random_state=42)
            search.fit(X_train, y_train)
            score = f1_score(y_test, search.best_estimator_.predict(X_test), average='macro')
            report = classification_report(y_test, search.best_estimator_.predict(X_test))
            logging.info(f"Model: {name}, F1-Score: {score}\n{report}")
            with open(f"train_log_{coin}.txt", "a") as f:
                f.write(f"{name}|{score}|{pd.Timestamp.now()}\n")
            if score > best_score:
                best_score, best_model, best_model_name = score, search.best_estimator_, name

        if best_model is not None:
            joblib.dump(best_model, f"trading_signal_model_{coin}.pkl")
            logging.info(f"Best model for {coin}: {best_model_name} with F1-Score: {best_score} saved!")
        else:
            logging.warning("No best model found. Possibly no valid training occurred.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
    end_time = time.time()
    logging.info(f"Training for {coin} took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for a specific coin")
    parser.add_argument("--coin", type=str, default="BTC", help="Coin symbol (e.g., BTC, ETH)")
    parser.add_argument("--threshold", type=float, default=0.002, help="Threshold for label creation")
    parser.add_argument("--once", action="store_true", help="Train only once without scheduling")
    args = parser.parse_args()

    if args.once:
        train_and_select_best_model(threshold=args.threshold, coin=args.coin)
    else:
        train_and_select_best_model(threshold=args.threshold, coin=args.coin)
        schedule.every().day.at("00:00").do(lambda: train_and_select_best_model(threshold=args.threshold, coin=args.coin))
        logging.info("⏳ Bot waiting for scheduled training...")
        while True:
            schedule.run_pending()
            time.sleep(60)
