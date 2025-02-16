# Trading Bot

Đây là dự án Trading Bot được xây dựng bằng Python với giao diện điều khiển thông qua Streamlit. Dự án cho phép:

- Huấn luyện mô hình dự đoán tín hiệu giao dịch (BUY, SELL, HOLD) cho từng coin riêng biệt (có thể chọn từ danh sách top 100 coin).
- Chạy bot giao dịch cho 1 đến 3 coin được chọn cùng lúc, với mỗi coin có mô hình huấn luyện riêng.
- Tích hợp tính năng lấy số dư tài khoản thực từ sàn KuCoin để phân bổ vốn giao dịch.
- Giao diện dashboard hiển thị kết quả huấn luyện (log, biểu đồ F1-Score) và kết quả giao dịch (profit, các lệnh, …).

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Các file chính](#các-file-chính)
- [Yêu cầu](#yêu-cầu)
- [Cách chạy dự án](#cách-chạy-dự-án)
- [Deploy trên Streamlit Cloud](#deploy-trên-streamlit-cloud)
- [Cấu hình biến môi trường (Secrets)](#cấu-hình-biến-môi-trường-secrets)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Liên hệ](#liên-hệ)

## Tính năng chính

1. **Huấn luyện mô hình**:  
   - Huấn luyện mô hình dự đoán tín hiệu giao dịch (BUY, SELL, HOLD) cho từng coin riêng (ví dụ: BTC, ETH, ADA, …).
   - Sử dụng GridSearchCV với các mô hình như RandomForest, XGBoost, LightGBM, và MLP.
   - Xử lý dữ liệu mất cân bằng bằng SMOTE và lưu log huấn luyện cho từng coin.

2. **Bot giao dịch đa coin**:  
   - Cho phép chọn từ 1 đến 3 coin từ danh sách top 100 coin (lấy từ CoinGecko).
   - Mỗi coin giao dịch dựa trên mô hình huấn luyện riêng.
   - Tích hợp tính năng lấy số dư tài khoản thực (USDT) từ KuCoin để tính khối lượng giao dịch phù hợp.
   - Đặt lệnh giao dịch và cập nhật trailing stop tự động.

3. **Dashboard Streamlit**:  
   - Giao diện điều khiển cho phép bắt đầu/dừng bot, huấn luyện mô hình và xem log kết quả huấn luyện cũng như kết quả giao dịch.
   - Cho phép chọn các coin giao dịch và hiển thị kết quả huấn luyện riêng cho từng coin.

## Cấu trúc dự án
	
trading-bot/
 ├── app_bot.py # Module chạy bot giao dịch đa coin
 ├── daily_train.py # Module huấn luyện mô hình cho từng coin
 ├── dashboard.py # Giao diện Streamlit để điều khiển bot và xem log
 ├── requirements.txt # Danh sách các thư viện cần cài đặt
 ├── README.md # Tài liệu dự án này


## Các file chính

- **app_bot.py**:  
  Chứa các hàm để lấy dữ liệu, thêm chỉ báo, dự đoán tín hiệu, tính toán khối lượng giao dịch, đặt lệnh và chạy bot cho nhiều coin cùng lúc. Chạy qua đối số dòng lệnh `--coins` để xác định các coin giao dịch.

- **daily_train.py**:  
  Module huấn luyện mô hình cho từng coin. Nó lấy dữ liệu từ KuCoin, thêm các chỉ báo kỹ thuật, tạo nhãn giao dịch, áp dụng SMOTE để cân bằng dữ liệu, huấn luyện các mô hình với GridSearchCV, và lưu mô hình cùng với label encoder cho coin đó. Log huấn luyện được lưu vào file `train_log_{coin}.txt`.

- **dashboard.py**:  
  Giao diện Streamlit cho phép người dùng:
  - Chọn từ 1 đến 3 coin để giao dịch (sử dụng multiselect).
  - Bắt đầu/dừng bot giao dịch (gọi app_bot.py qua subprocess).
  - Huấn luyện mô hình cho từng coin được chọn (gọi daily_train.py qua subprocess).
  - Hiển thị log huấn luyện, biểu đồ hiệu suất, và kết quả giao dịch (profit log).

## Yêu cầu

- Python 3.8+
- Các thư viện được liệt kê trong file **requirements.txt**.
- Tài khoản API của KuCoin (cấu hình qua secrets).
- Kết nối Internet để lấy dữ liệu từ CoinGecko và API của KuCoin.

## Cách chạy dự án

### Chạy local
1. Cài đặt các gói cần thiết:
   ```bash
   pip install -r requirements.txt
2. Chạy dashboard:
   streamlit run dashboard.py
3. Trong dashboard, bạn có thể chọn các coin, huấn luyện mô hình và bắt đầu bot giao dịch.
Chạy từng module
	Huấn luyện mô hình cho một coin:
	'''bash
	python daily_train.py --coin BTC --once

Chạy bot giao dịch cho nhiều coin:
	'''bash
	python app_bot.py --coins BTC,ETH,ADA,...

## Deploy trên Streamlit Cloud
	Tạo repository trên GitHub với toàn bộ mã nguồn dự án (bao gồm app_bot.py, daily_train.py, dashboard.py, requirements.txt, README.md, và thư mục .streamlit nếu cần secrets).
	Đăng nhập vào Streamlit Cloud tại share.streamlit.io bằng tài khoản GitHub của bạn.
	Tạo app mới:
	Chọn repository, branch (thường là main), và file chính (dashboard.py).
	Cấu hình biến môi trường (Secrets):
	Qua giao diện Secrets của Streamlit Cloud, thêm các API keys cần thiết.
	Deploy và kiểm tra:
	Sau khi deploy, ứng dụng sẽ có URL công khai. Bạn có thể truy cập và kiểm tra trên điện thoại hoặc trình duyệt.	
## Cấu hình biến môi trường (Secrets)
	Nếu bạn sử dụng biến môi trường cho API keys, hãy cấu hình file .streamlit/secrets.toml hoặc qua giao diện của Streamlit Cloud:
	KUCOIN_API_KEY = "your_api_key"
	KUCOIN_SECRET = "your_secret"
	KUCOIN_PASSPHRASE = "your_passphrase"
	TELEGRAM_TOKEN = "your_telegram_bot_token"
	CHAT_ID = "your_chat_id"
## Hướng dẫn sử dụng Dashboard
	Chọn coin để trading:
	Sử dụng widget multiselect trong sidebar để chọn từ 1 đến 3 coin (danh sách được lấy từ CoinGecko và lọc với danh sách coin mà KuCoin hỗ trợ).
	Bắt đầu Bot:
	Nhấn nút "Start Bot" để khởi chạy giao dịch cho các coin đã chọn.
	Huấn luyện mô hình:
	Nhấn nút "Train Model" để huấn luyện mô hình cho từng coin được chọn. Kết quả huấn luyện và biểu đồ hiệu suất sẽ được hiển thị riêng cho từng coin.
	Theo dõi kết quả giao dịch:
	Dashboard hiển thị log giao dịch, lợi nhuận và biểu đồ tổng hợp.
## Liên hệ
	Nếu có bất kỳ câu hỏi hoặc gặp khó khăn, hãy liên hệ qua email hoặc mở issue trên GitHub.

