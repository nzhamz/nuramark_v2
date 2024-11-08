import streamlit as st
import yfinance as yf
import backtrader as bt
import math
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# GitHub link for CSV file
csv_url = "https://raw.githubusercontent.com/nzhamz/nuramark_v2/main/Trade.csv"

# Streamlit app title and instructions


# Streamlit app title
st.title("nzham 📈 Trading Strategy Performance")

# Download and load data
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# Load the data
data = load_data(csv_url)

# Display the data
st.write("### Optimized Trading Data")
st.dataframe(data)



st.write("Customize the parameters and view the trading strategy outcome:")

# Collect user input
with st.form("input_form"):
    src = st.text_input("Source (Ticker Symbol, Crypto: XXX-USD, TASI: XXX.SR )", "NEAR-USD")
    interval = st.selectbox("Interval", options=["1h", "1d", "1w"], index=1)
    start_date = st.date_input("Start Date", value=datetime(2024, 6, 1).date())
    end_date = st.date_input("End Date", value=datetime.now().date())
    
    buy_ma_period = st.number_input("Buy Moving Average Period", min_value=1, max_value=100, value=10)
    sell_ma_period = st.number_input("Sell Moving Average Period", min_value=1, max_value=100, value=10)
    buy_slope = st.number_input("Buy Slope", min_value=-10.0, max_value=10.0, value=2.0)
    sell_slope = st.number_input("Sell Slope", min_value=-10.0, max_value=10.0, value=-2.0)
    wait_period = st.number_input("Wait Period", min_value=1, max_value=10, value=3)
    hold_period = st.number_input("Hold Period", min_value=1, max_value=30, value=4)
    profit_target = st.number_input("Profit Target Percentage", min_value=0.1, max_value=100.0, value=5.0)
    stop_loss = st.number_input("Stop Loss Percentage", min_value=-100.0, max_value=0.0, value=-5.0)
    
    submitted = st.form_submit_button("Run Strategy")

# Only execute the strategy if the form is submitted
if submitted:
    data = yf.download(src, start=start_date, end=end_date, interval=interval)
    
    class PandasData(bt.feeds.PandasData):
        lines = ('low',)
        params = (('low', -1),)

    data_feed = PandasData(dataname=data)
    
    class EMASlopeStrategy(bt.Strategy):
        params = dict(
            buy_ma_period=buy_ma_period, sell_ma_period=sell_ma_period,
            buy_slope=buy_slope, sell_slope=sell_slope,
            wait_period=wait_period, hold_period=hold_period,
            profit_target_percentage=profit_target, stop_loss_percentage=stop_loss
        )
        
        def __init__(self):
            self.buy_ema = bt.indicators.ExponentialMovingAverage(self.data.low, period=self.params.buy_ma_period)
            self.sell_ema = bt.indicators.ExponentialMovingAverage(self.data.low, period=self.params.sell_ma_period)
            self.buy_variable = 0
            self.sell_variable = 0
            self.trades = []
            self.profit = 0
            self.winning_trades = 0
            self.total_trades = 0
            self.entry_price = None

        def calc_slope(self, ema):
            if len(ema) > 1:
                segment_size = 1
                slope = ((ema[0] - ema[-segment_size]) / ema[-segment_size]) * 100 if ema[-segment_size] != 0 else 0
                return math.atan(slope) * 180 / math.pi
            return 0

        def next(self):
            buy_slope = self.calc_slope(self.buy_ema)
            sell_slope = self.calc_slope(self.sell_ema)

            if buy_slope > self.params.buy_slope:
                self.buy_variable = 1
            else:
                self.buy_variable = 0

            if sell_slope < self.params.sell_slope:
                self.sell_variable = 1
            else:
                self.sell_variable = 0

            if self.buy_variable == 1 and not self.position:
                self.buy_price = self.data.open[0]
                self.entry_price = self.buy_price
                self.buy()
                self.trades.append((self.data.datetime.date(0), "Buy", self.buy_price))

            elif self.position:
                current_price = self.data.open[0]
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100

                if profit_pct >= self.params.profit_target_percentage:
                    self.sell()
                    self.profit += profit_pct
                    self.winning_trades += 1
                    self.total_trades += 1
                    self.trades.append((self.data.datetime.datetime(0), "Sell", current_price, profit_pct))
                    self.entry_price = None

                elif profit_pct <= self.params.stop_loss_percentage:
                    self.sell()
                    self.profit += profit_pct
                    self.total_trades += 1
                    self.trades.append((self.data.datetime.datetime(0), "Sell", current_price, profit_pct))
                    self.entry_price = None

        def get_performance(self):
            total_trades = self.total_trades
            winning_trades = self.winning_trades
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            return total_trades, winning_trades, accuracy, self.profit

        def plot_trades(self):
            buy_dates = [t[0] for t in self.trades if t[1] == "Buy"]
            sell_dates = [t[0] for t in self.trades if t[1] == "Sell"]
            buy_prices = [t[2] for t in self.trades if t[1] == "Buy"]
            sell_prices = [t[2] for t in self.trades if t[1] == "Sell"]
            profits = [t[3] for t in self.trades if t[1] == "Sell"]
        
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(data.index, data['Close'], label='Close Price', color='blue')
            ax.scatter(data.index, data['Close'], color='black', s=30, alpha=0.7)
            ax.scatter(buy_dates, buy_prices, marker='o', color='green', label='Buy Signal', s=100)
            ax.scatter(sell_dates, sell_prices, marker='X', color='red', label='Sell Signal', s=100)
        
            for i, txt in enumerate(profits):
                ax.annotate(f"{txt:.2f}%", (sell_dates[i], sell_prices[i]), textcoords="offset points", xytext=(0,10), ha='center')

            params_text = (
                f"Source (Ticker): {src}\n"
                f"Interval: {interval}\n"
                f"Buy EMA Period: {self.params.buy_ma_period}\n"
                f"Sell EMA Period: {self.params.sell_ma_period}\n"
                f"Profit Target: {self.params.profit_target_percentage}%\n"
                f"Stop Loss: {self.params.stop_loss_percentage}%"
            )
            ax.text(0.02, 0.95, params_text, transform=ax.transAxes, fontsize=9, va='top', color='black', bbox=dict(boxstyle="round,pad=0.6", edgecolor='gray', facecolor='white', alpha=0.7))
            ax.set_title('Trading Strategy Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMASlopeStrategy)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000)
    strategy_instance = cerebro.run()[0]

    total_trades, winning_trades, accuracy, total_profit = strategy_instance.get_performance()
    st.write("### Strategy Performance")
    st.write(f"Total Trades: {total_trades}")
    st.write(f"Winning Trades: {winning_trades}")
    st.write(f"Accuracy: {accuracy:.2f}%")
    st.write(f"Total Profit: {total_profit:.2f}%")
    strategy_instance.plot_trades()
    
    if strategy_instance.trades:
        last_trade_type = strategy_instance.trades[-1][1]
        if last_trade_type == "Buy":
            st.markdown("<h2 style='color: green;'>📈 Buy Recommendation</h2>", unsafe_allow_html=True)
        elif last_trade_type == "Sell":
            st.markdown("<h2 style='color: red;'>📉 Sell Recommendation</h2>", unsafe_allow_html=True)
    else:
        st.write("No trades executed.")
        


