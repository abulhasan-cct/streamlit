import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from binance.client import Client
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import time
import requests
# ---------------------------
# Logging Setup in Session State
# ---------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(message):
    st.session_state.logs.append(message)
    print(message)  # Also print to console (optional)
    # send_telegram_message(message)  # Send log to Telegram

# Set the ticker symbol for the asset (Example: 'DOGE-USD' for Dogecoin)
ticker = "DOGE-USD"  # Change this to the asset you want to track
# ---------------------------
# Binance Testnet Setup
# ---------------------------
# IMPORTANT: Do not hardcode API keys in production!
BINANCE_API_KEY = "iwz7z9QAM8hfxCfFzTRaaj9UrfrQLBTKQHuwM8hAhX1u6EwoJlYbrHEnjoqlKGR8"
BINANCE_SECRET_KEY = "iVbpaANVrvsAzd546oRdLx9AiSRhdBi5fFEkZRxQdd5EsotNvxwP69lWD5REdDrQ"

client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY, testnet=True)

# ---------------------------
# Indicator & Signal Functions
# ---------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(data, sma_short=50, sma_long=200, rsi_period=14):
    log("Computing indicators...")
    data['SMA_short'] = data['Close'].rolling(window=sma_short, min_periods=1).mean()
    data['SMA_long'] = data['Close'].rolling(window=sma_long, min_periods=1).mean()
    data['RSI'] = compute_RSI(data['Close'], rsi_period)
    return data

def generate_signals(data):
    log("Generating signals...")
    data = data.copy()
    data['Signal'] = 0
    for i in range(1, len(data)):
        # Buy when short SMA is above long SMA and RSI is below 30
        if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i] and data['RSI'].iloc[i] < 60:
            data.at[data.index[i], 'Signal'] = 1
        # Sell when short SMA is below long SMA or RSI is above 70
        elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i] or data['RSI'].iloc[i] > 70:
            data.at[data.index[i], 'Signal'] = -1
    return data

TELEGRAM_BOT_TOKEN = "7340249741:AAG6nI7bM0Pnwt29AzBXisiRQpRSOExQML0"
TELEGRAM_CHAT_ID = "1032676639"

def send_telegram_message(message):
    """Send a message to the Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram message sent successfully!")
        else:
            print(f"Error sending message: {response.text}")
    except Exception as e:
        print(f"Telegram Error: {e}")

def backtest_and_place_orders(data, initial_capital=100):
    log("Starting backtest and placing orders...")

    capital = initial_capital
    position = 0
    trades = []
    open_trades = []
    closed_trades = []
    data = data.copy()
    data['Position'] = 0
    data['Signal'] = pd.to_numeric(data['Signal'], errors='coerce')

    for i in range(1, len(data)):
        signal_val = data['Signal'].iloc[i]

        if isinstance(signal_val, (int, float)):
            if signal_val == 1 and position == 0:
                buy_price = data['Open'].iloc[i]
                position = float(capital / buy_price)
                capital = 0
                trades.append(('Buy', data.index[i], buy_price, position))
                open_trades.append({'Buy': data.index[i], 'Price': buy_price, 'Position': position})
                data.at[data.index[i], 'Position'] = position
                log(f"Buy executed at {data.index[i]} | Price: {buy_price:.2f} | Qty: {position:.4f}")
                
                send_telegram_message(f"📈 Buy Order Executed\n🕒 {data.index[i]}\n💰 Price: {buy_price:.2f}\n🔢 Quantity: {position:.4f}")

            elif signal_val == -1 and position > 0:
                sell_price = data['Open'].iloc[i]
                capital = position * sell_price
                trades.append(('Sell', data.index[i], sell_price, position))
                position = 0
                if open_trades:
                    open_trade = open_trades.pop()
                    open_trade.update({'Sell': data.index[i], 'Sell Price': sell_price})
                    closed_trades.append(open_trade)
                data.at[data.index[i], 'Position'] = 0
                log(f"Sell executed at {data.index[i]} | Price: {sell_price:.2f}")

                send_telegram_message(f"📉 Sell Order Executed\n🕒 {data.index[i]}\n💰 Price: {sell_price:.2f}\n🔢 Quantity: {position:.4f}")

    return capital, trades, data, open_trades, closed_trades


# ---------------------------
# REAL TIME ASSTES CONVERTION
# ---------------------------
def get_asset_value(asset_name):
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={asset_name}USDT'
    response = requests.get(url)
    data = response.json()
    if 'price' in data:
        return float(data['price'])
    return None

# Function to get the USDT to INR conversion rate
def get_usdt_inr_rate():
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=inr'
    response = requests.get(url)
    data = response.json()
    if 'tether' in data and 'inr' in data['tether']:
        return float(data['tether']['inr'])
    return None

# Streamlit UI elements
st.title("Real-time Asset Value and INR Conversion")

# List of assets (can be expanded with more assets)
asset_list = ["DOGE", "BTC", "ETH", "ADA", "SOL"]
selected_asset = st.selectbox("Select Asset", asset_list)

if selected_asset:
    # Get real-time asset value
    asset_value_usdt = get_asset_value(selected_asset)
    
    if asset_value_usdt is not None:
        # Get USDT to INR conversion rate
        usdt_inr_rate = get_usdt_inr_rate()
        
        if usdt_inr_rate is not None:
            # Convert asset value to INR
            asset_value_inr = asset_value_usdt * usdt_inr_rate
            
            # Display the real-time value and INR conversion
            st.write(f"**{selected_asset} Value:** {asset_value_usdt} USDT")
            st.write(f"**{selected_asset} Value in INR:** ₹{asset_value_inr:,.2f}")
        else:
            st.write("Error fetching USDT to INR conversion rate.")
    else:
        st.write(f"Error fetching {selected_asset} value.")
# ---------------------------
# Binance Order & Account Functions
# ---------------------------
def place_binance_order(symbol, side, quantity):
    try:
        # Placing an order on the Binance testnet
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity
        )
        return order
    except Exception as e:
        log(f"Error placing order: {str(e)}")
        return None

def get_balance():
    log("Fetching balance from Binance...")
    try:
        balance = client.get_asset_balance(asset='USDT')
        return balance
    except Exception as e:
        st.error(f"Error fetching balance: {e}")
        log(f"Error fetching balance: {e}")
        return None

def get_trading_results():
    log("Fetching trading results from Binance...")
    try:
        orders = client.get_all_orders(symbol='DOGEUSDT')
        return orders
    except Exception as e:
        st.error(f"Error fetching trading results: {e}")
        log(f"Error fetching trading results: {e}")
        return None

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Quant Trading Bot: RSI, SMA & Binance Testnet")
    st.markdown("""
    This bot uses a quantitative strategy combining SMA and RSI indicators.
    It fetches historical data via **yfinance**, backtests the strategy, and simulates paper trading on Binance Testnet.
    """)

    # Auto-refresh every 30 seconds
    st_autorefresh(interval=30000, limit=None, key="trade_refresh")

    # ---------------------------
    # Asset & Period Selection
    # ---------------------------
    asset_options = {
        "Dogecoin (DOGE-USD)": "DOGE-USD",
        "Gold (GLD ETF)": "GLD",
        "Apple (AAPL)": "AAPL",
        "Nvidia (NVDA)": "NVDA"
    }
    asset_name = st.sidebar.selectbox("Select Asset", list(asset_options.keys()))
    ticker = asset_options[asset_name]

    period = st.sidebar.selectbox("Select Data Period", ["1y", "2y", "5y"])
    st.sidebar.write(f"Fetching data for **{asset_name}** over the past **{period}**.")
    log(f"Selected asset: {asset_name} ({ticker}), Period: {period}")

    # ---------------------------
    # Data Fetching & Analysis
    # ---------------------------
    data = yf.download(ticker, period=period)
    if data.empty:
        st.error("No data fetched. Please check the ticker or your network connection.")
        log("Data fetch failed.")
        return
    log("Historical data fetched successfully.")

    data = compute_indicators(data)
    data = generate_signals(data)
    final_value, trades, data, open_trades, closed_trades = backtest_and_place_orders(data)

    st.subheader("Backtest Results")
    if final_value is not None:
        st.write(f"**Final Portfolio Value:** ${final_value:,.2f}")
    else:
        st.write("**Trade is still open.**")

    if trades:
        trade_df = pd.DataFrame(trades, columns=["Type", "Date", "Price", "Quantity"])
        trade_df["Date"] = trade_df["Date"].dt.date
        st.write("**Trades Executed:**")
        st.dataframe(trade_df)
    
    # ---------------------------
    # Real-Time P/L for Open Trades
    # ---------------------------
    st.subheader("Real-Time Open Trade P/L")
    if open_trades:
        try:
            # Fetch current price using a 1-minute interval (for near real-time data)
            current_data = yf.download(ticker, period="1d", interval="1m")
            current_price = current_data['Close'].iloc[-1]
            log(f"Current price fetched: {current_price:.2f}")
        except Exception as e:
            st.error("Error fetching current price. Using last available price.")
            log(f"Error fetching current price: {e}")
            current_price = data['Close'].iloc[-1]
        
        pl_data = []
        for trade in open_trades:
            buy_price = trade['Price']
            quantity = trade['Position']
            pl = (current_price - buy_price) * quantity
            pl_data.append({
                "Buy Time": trade['Buy'].date(),
                "Buy Price": buy_price,
                "Quantity": quantity,
                "Current Price": current_price,
                "Unrealized P/L": pl
            })
        pl_df = pd.DataFrame(pl_data)
        st.dataframe(pl_df)
    else:
        st.write("No open trades at the moment.")

    # ---------------------------
    # Price & Indicator Charts
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label="Close Price", color="blue", alpha=0.8)
    ax.plot(data.index, data['SMA_short'], label="SMA (50)", linestyle="--", color="orange")
    ax.plot(data.index, data['SMA_long'], label="SMA (200)", linestyle="--", color="red")
    buy_signals = data[data['Signal'] == 1]
    ax.scatter(buy_signals.index, buy_signals['Close'], marker="^", color="green", label="Buy Signal", s=100)
    sell_signals = data[data['Signal'] == -1]
    ax.scatter(sell_signals.index, sell_signals['Close'], marker="v", color="red", label="Sell Signal", s=100)
    ax.set_title(f"{asset_name} Price Chart with SMA & Buy/Sell Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    plt.xticks(rotation=45)
    st.pyplot(fig)


    def check_open_orders(symbol):
        open_orders = client.get_open_orders(symbol=symbol)
        return open_orders

    # Test the check for open orders
    orders = check_open_orders("DOGEUSDT")
    print(f"Open Orders: {orders}")
    # ---------------------------
    # RSI Chart
    # ---------------------------
    # st.subheader("RSI Indicator Chart")
    # fig2, ax2 = plt.subplots(figsize=(12, 4))
    # ax2.plot(data.index, data['RSI'], label="RSI", color="purple")
    # ax2.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    # ax2.axhline(70, linestyle="--", color="red", label="Overbought (70)")
    # ax2.set_title("RSI Indicator")
    # ax2.set_xlabel("Date")
    # ax2.set_ylabel("RSI")
    # ax2.legend(loc="best")
    # plt.xticks(rotation=45)
    # st.pyplot(fig2)
    

    # ---------------------------
    # Display Process Logs in the UI
    # ---------------------------
    st.subheader("Process Logs")
    log_text = "\n".join(st.session_state.logs[-20:])  # Show the last 20 log messages
    st.text_area("Logs", log_text, height=200)

    # ---------------------------
    # Binance Trading Functions
    # ---------------------------
    st.subheader("Binance Testnet Trading Results")
    balance = get_balance()
    if balance:
        st.write(f"**Balance (USDT):** {balance['free']} USDT")

    orders = get_trading_results()
    if orders:
        # Convert the orders list/dict into a DataFrame for nicer display
        orders_df = pd.DataFrame(orders)
        st.write("**Trading History:**")
        st.dataframe(orders_df)
    else:
        st.write("No trading history available from Binance.")

# ---------------------------
# Wallet Balances
# ---------------------------
st.subheader("Wallet Balances")

# Initialize balances to ensure they exist in the local scope
usdt_balance = None
doge_balance = None

try:
    usdt_balance = get_balance()
except Exception as e:
    st.error(f"Error fetching USDT balance: {e}")
    log(f"Error fetching USDT balance: {e}")

try:
    doge_balance = client.get_asset_balance(asset="DOGE")
except Exception as e:
    st.error(f"Error fetching DOGE balance: {e}")
    log(f"Error fetching DOGE balance: {e}")

if usdt_balance:
    st.write(f"**USDT Balance:** {usdt_balance['free']} USDT")
else:
    st.write("USDT balance not available.")

if doge_balance:
    st.write(f"**DOGE Balance:** {doge_balance['free']} DOGE")
else:
    st.write("DOGE balance not available.")

# ---------------------------
# Manual Buy Option to Open a Trade
# ---------------------------
st.subheader("Manual Buy Option")
if usdt_balance:
    try:
        available_usdt = float(usdt_balance['free'])
    except (KeyError, ValueError) as e:
        st.error("Error reading USDT balance value.")
        log(f"Error reading USDT balance value: {e}")
        available_usdt = 0

    default_buy_amount = 100.00 # default to 50% of available USDT
    buy_amount = st.number_input("Enter USDT amount to use for buying DOGE:", 
                                 min_value=0.0, 
                                 value=default_buy_amount, 
                                 step=1.0)
    if st.button("Place Buy Order"):
        try:
            # Fetch current DOGE price from Binance
            ticker_info = client.get_symbol_ticker(symbol="DOGEUSDT")
            current_price = float(ticker_info['price'])
            # Calculate the raw quantity of DOGE to buy
            raw_quantity = buy_amount / current_price
            # For DOGE, Binance might require an integer quantity. Truncate to an integer.
            quantity = int(raw_quantity)
            
            log(f"Placing BUY order: Spending {buy_amount} USDT to buy {quantity} DOGE at price {current_price:.4f}")
            order_response = place_binance_order("DOGEUSDT", "BUY", quantity)
            if order_response:
                st.success("Buy order placed successfully!")
                log(f"Buy order placed for {quantity} DOGEUSDT at market price.")

                # Send Telegram Notification
                send_telegram_message(f"✅ BUY ORDER PLACED\n💰 {buy_amount} USDT spent\n🔹 {quantity} DOGE bought\n📈 Price: {current_price:.4f}")

            else:
                st.error("Buy order failed. Check logs for details.")
        except Exception as e:
            st.error(f"Error during buy order: {e}")
            log(f"Error during buy order: {e}")
else:
    st.write("USDT balance not available, cannot place buy order.")


# ---------------------------
# Manual Sell Option with Quantity Input
# ---------------------------
st.subheader("Manual Sell Option")
if doge_balance:
    try:
        available_doge = float(doge_balance['free'])
    except (KeyError, ValueError) as e:
        st.error("Error reading DOGE balance value.")
        log(f"Error reading DOGE balance value: {e}")
        available_doge = 0

    st.write(f"You currently hold **{available_doge} DOGE**.")
    
    # Allow user to specify how much DOGE to sell
    sell_quantity = st.number_input(
        "Enter quantity of DOGE to sell:",
        min_value=0.0,
        max_value=available_doge,
        value=0.0,
        step=1.0,  # Use step=0.1 if decimals are allowed; for DOGE, usually whole numbers are used.
    )
    
    if st.button("Sell DOGE"):
        try:
            if sell_quantity > 0:
                # Adjust the quantity precision as required by Binance. Here we assume whole numbers.
                quantity = int(sell_quantity)
                log(f"Placing SELL order for {quantity} DOGE at market price.")
                order_response = place_binance_order("DOGEUSDT", "SELL", quantity)
                if order_response:
                    st.success("Sell order placed successfully!")
                    log(f"Sell order placed for {quantity} DOGEUSDT at market price.")

                    # Send Telegram Notification
                    send_telegram_message(f"❌ SELL ORDER PLACED\n🔹 {quantity} DOGE sold\n📉 Market Price Order Executed")

                else:
                    st.error("Sell order failed. Check logs for details.")
            else:
                st.error("Quantity to sell must be greater than zero.")
        except Exception as e:
            st.error(f"Error during sell order: {e}")
            log(f"Error during sell order: {e}")
else:
    st.write("DOGE balance not available.")


# ---------------------------
# All Time Profit & Loss Calculation
# ---------------------------
st.subheader("All Time Profit and Loss")

# Let the user enter their initial capital (if not hardcoded)
initial_capital = st.number_input("Enter your initial capital (USDT):", value=10000.0, step=100.0)

# Get current USDT balance
if usdt_balance:
    try:
        current_usdt = float(usdt_balance['free'])
    except Exception as e:
        st.error("Error reading USDT balance value.")
        log(f"Error reading USDT balance value: {e}")
        current_usdt = 0
else:
    current_usdt = 0

# Get current DOGE balance
if doge_balance:
    try:
        current_doge = float(doge_balance['free'])
    except Exception as e:
        st.error("Error reading DOGE balance value.")
        log(f"Error reading DOGE balance value: {e}")
        current_doge = 0
else:
    current_doge = 0

# Fetch current DOGE price from Binance
try:
    ticker_info = client.get_symbol_ticker(symbol="DOGEUSDT")
    current_doge_price = float(ticker_info['price'])
    log(f"Current DOGE price fetched: {current_doge_price}")
except Exception as e:
    st.error(f"Error fetching current DOGE price: {e}")
    log(f"Error fetching current DOGE price: {e}")
    current_doge_price = 0

# Calculate current equity: USDT balance + (DOGE balance * current DOGE price)
current_equity = current_usdt + (current_doge * current_doge_price)

# Calculate all time profit/loss
profit_loss = current_equity - initial_capital

st.write(f"**Current Account Equity:** {current_equity:.2f} USDT")
st.write(f"**All Time Profit/Loss:** {profit_loss:.2f} USDT")


# ---------------------------
# MAIN FUNCTION START
# ---------------------------

if __name__ == "__main__":
    main()
# ---------------------------
# MAIN FUNCTION END
# ---------------------------

# ---------------------------
# RSI REAL TIME DATA
# ---------------------------
# Define thresholds and global tracking variables
BUY_THRESHOLD = 30  # RSI below 30 indicates an oversold condition
SELL_THRESHOLD = 70  # RSI above 70 indicates an overbought condition
RSI_CHANGE_THRESHOLD = 5  # Minimum RSI change required to trigger a new alert
PRICE_CHANGE_THRESHOLD = 3  # Minimum price change (%) required to trigger an alert

last_rsi = None
last_price = None

st.subheader(f"📈 Real-Time RSI Indicator Chart for {ticker}")

def calculate_rsi(data, period=14):
    """Calculate RSI for the given data."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def fetch_data():
    """Fetch real-time stock data and calculate RSI."""
    try:
        data = yf.download(ticker, period="5d", interval="5m")
        data = calculate_rsi(data)
        return data.tail(50)
    except Exception as e:
        st.error(f"❌ Error fetching real-time data: {e}")
        return None

chart_placeholder = st.empty()

while True:
    realtime_data = fetch_data()
    if realtime_data is not None:
        latest_price = realtime_data['Close'].iloc[-1].item()
        latest_rsi = realtime_data['RSI'].iloc[-1].item()


        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=realtime_data.index, y=realtime_data['Close'], mode='lines', name='Price', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=realtime_data.index, y=realtime_data['RSI'], mode='lines', name='RSI (14)', line=dict(color='orange', width=2), yaxis="y2"))
        fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), annotation_text="Overbought")
        fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), annotation_text="Oversold")
        fig.update_layout(
            title=f"Live RSI Indicator for {ticker}",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
            template="plotly_dark",
            transition={'duration': 500, 'easing': 'cubic-in-out'}
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")

        # Check significant differences before sending alerts
        if last_rsi is not None and last_price is not None:
            rsi_change = abs(latest_rsi - last_rsi)
            price_change_percent = abs((latest_price - last_price) / last_price) * 100

            if latest_rsi < BUY_THRESHOLD and rsi_change >= RSI_CHANGE_THRESHOLD:
                send_telegram_message(f"📉 RSI Alert: {latest_rsi:.2f} (BUY Signal). Price: ${latest_price:.2f}")
                last_rsi = latest_rsi

            elif latest_rsi > SELL_THRESHOLD and rsi_change >= RSI_CHANGE_THRESHOLD:
                send_telegram_message(f"📈 RSI Alert: {latest_rsi:.2f} (SELL Signal). Price: ${latest_price:.2f}")
                last_rsi = latest_rsi

            if price_change_percent >= PRICE_CHANGE_THRESHOLD:
                send_telegram_message(f"⚠️ Price Alert: ${latest_price:.2f} (Change: {price_change_percent:.2f}%)")
                last_price = latest_price

        # Store initial values if not set
        if last_rsi is None:
            last_rsi = latest_rsi
        if last_price is None:
            last_price = latest_price

    time.sleep(30)  # Check every 30 seconds