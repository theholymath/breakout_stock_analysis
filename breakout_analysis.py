import pandas as pd
import numpy as np
import yfinance as yf
from pytickersymbols import PyTickerSymbols
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pytz
import warnings

warnings.filterwarnings("ignore")


def get_nasdaq_100_tickers():
    stock_data = PyTickerSymbols()
    nasdaq_tickers = stock_data.get_stocks_by_index(
        "NASDAQ 100"
    )  # Corrected index name
    return [stock["symbol"] for stock in nasdaq_tickers]


def fetch_stock_data(
    ticker,
    start_date,
    end_date,
    volume_threshold=200.0,
    price_threshold=2.0,
    holding_period=10,
):
    """
    Fetches and processes historical stock data to identify breakout days and calculate returns.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (datetime): Start date for data fetching.
        end_date (datetime): End date for data fetching.
        volume_threshold (float): % above the 20-day average volume to qualify as breakout.
        price_threshold (float): % increase in closing price to qualify as breakout.
        holding_period (int): # of days to hold the stock after a breakout.

    Returns:
        pd.DataFrame: DataFrame containing breakout signals and buy/sell info.
    """
    # --------------------
    # 1. Fetch raw data
    # --------------------
    ticker_obj = yf.Ticker(ticker)
    buffer_days = 60
    fetch_start = start_date - timedelta(days=buffer_days)

    # yfinance end date is exclusive, so add 1 day to capture the entire end_date
    raw_data = ticker_obj.history(start=fetch_start, end=end_date + timedelta(days=1))

    # Handle empty data
    if raw_data.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return pd.DataFrame()

    # --------------------
    # 2. Prep the DataFrame
    # --------------------
    data = raw_data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])  # Ensure Date is a datetime
    data.sort_values("Date", inplace=True)
    data.reset_index(drop=True, inplace=True)

    # If there's a timezone, remove it
    if data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_convert(None)

    # Make sure required columns exist
    required = ["Date", "Close", "Volume"]
    if not all(col in data.columns for col in required):
        print(f"Missing required columns in {ticker} data: {data.columns.tolist()}")
        return pd.DataFrame()

    data = data[required]

    # --------------------
    # 3. 20-day average volume
    # --------------------
    data["Avg_Volume_20"] = (
        data["Volume"].astype(float).rolling(window=20, min_periods=20).mean().round(2)
    )

    # If no valid rolling volume, skip
    if data["Avg_Volume_20"].notna().sum() == 0:
        print(f"No 20-day average volume found for {ticker}.")
        return pd.DataFrame()

    # Drop rows with NaN in 'Avg_Volume_20'
    data.dropna(subset=["Avg_Volume_20"], inplace=True)

    # *** Important! ***
    # Reset index so that .iterrows() matches data.at[idx, ...] or data.loc[idx, ...]
    data.reset_index(drop=True, inplace=True)

    # --------------------
    # 4. Volume% & Price% Change
    # --------------------
    data["Volume_Pct"] = (data["Volume"] / data["Avg_Volume_20"]) * 100
    data["Price_Change_Pct"] = data["Close"].pct_change() * 100
    data["Price_Change_Pct"].fillna(0, inplace=True)  # first row fill w/ 0

    # --------------------
    # 5. Breakout column
    # --------------------
    data["Breakout"] = np.where(
        (data["Volume_Pct"] >= volume_threshold)
        & (data["Price_Change_Pct"] >= price_threshold),
        1,
        0,
    )

    # --------------------
    # 6. Identify breakout days in user date range
    # --------------------
    # Same logic as your Streamlit code:
    # (data["Date"] >= start_date) & (data["Date"] <= end_date)
    # If your start_date/end_date are naive datetimes, no tz conversion is needed
    breakout_days = data[
        (data["Breakout"] == 1)
        & (data["Date"] >= pd.to_datetime(start_date))
        & (data["Date"] <= pd.to_datetime(end_date))
    ]

    # --------------------
    # 7. Initialize Buy/Sell columns
    # --------------------
    data["Buy Date"] = np.nan
    data["Buy Price"] = np.nan
    data["Sell Date"] = np.nan
    data["Sell Price"] = np.nan
    data["Return (%)"] = np.nan

    # --------------------
    # 8. Loop over breakout days
    # --------------------
    for idx, row in breakout_days.iterrows():
        buy_date = row["Date"]
        buy_price = row["Close"]
        target_sell_date = buy_date + timedelta(days=holding_period)

        # Get the earliest row on or after target_sell_date
        sell_candidates = data[data["Date"] >= target_sell_date]
        if sell_candidates.empty:
            continue

        sell_date = sell_candidates.iloc[0]["Date"]
        sell_price = sell_candidates.iloc[0]["Close"]
        return_pct = ((sell_price - buy_price) / buy_price) * 100

        data.at[idx, "Buy Date"] = buy_date.strftime("%Y-%m-%d")
        data.at[idx, "Buy Price"] = round(buy_price, 2)
        data.at[idx, "Sell Date"] = sell_date.strftime("%Y-%m-%d")
        data.at[idx, "Sell Price"] = round(sell_price, 2)
        data.at[idx, "Return (%)"] = round(return_pct, 2)

    # --------------------
    # 9. Final columns
    # --------------------
    report_cols = [
        "Date",
        "Close",
        "Volume",
        "Avg_Volume_20",
        "Volume_Pct",
        "Price_Change_Pct",
        "Breakout",
        "Buy Date",
        "Buy Price",
        "Sell Date",
        "Sell Price",
        "Return (%)",
    ]
    for col in report_cols:
        if col not in data.columns:
            data[col] = np.nan

    return data[report_cols]


def analyze_aggregate_breakouts(list_of_dfs):
    """
    Given a list of DataFrames (one per ticker) where each DataFrame includes
    'Breakout' and 'Return (%)' columns, compute the overall statistics
    for *all* breakouts across all tickers.

    Args:
        list_of_dfs (list): A list of pandas DataFrames, each containing columns
                            including at least ['Breakout', 'Return (%)'].

    Returns:
        dict: A dictionary of aggregated statistics, e.g. average breakout return,
              median breakout return, min, max, etc.
    """
    # Keep a list of all returns from all breakouts across all tickers
    all_breakout_returns = []

    # Loop through each ticker's DataFrame
    for df in list_of_dfs:
        # Safety check: ensure 'Breakout' and 'Return (%)' exist
        if not {"Breakout", "Return (%)"}.issubset(df.columns):
            print("Skipping a DataFrame that doesn't have the required columns.")
            continue

        # Filter rows where Breakout == 1
        breakout_rows = df[df["Breakout"] == 1]

        # Some rows might not have a valid Return (%) if no Sell Date was found
        valid_returns = breakout_rows["Return (%)"].dropna()

        # Extend our master list
        all_breakout_returns.extend(valid_returns.tolist())

    # If we never found any breakout returns, just exit
    if len(all_breakout_returns) == 0:
        print("No breakout trades found among the data provided.")
        return {}

    # Convert to numpy array for easier stats
    all_breakout_returns = np.array(all_breakout_returns)

    # Compute some basic statistics
    avg_return = np.mean(all_breakout_returns)
    median_return = np.median(all_breakout_returns)
    max_return = np.max(all_breakout_returns)
    min_return = np.min(all_breakout_returns)

    # Format or round them as needed
    results = {
        "count_breakout_trades": len(all_breakout_returns),
        "average_return_%": round(avg_return, 2),
        "median_return_%": round(median_return, 2),
        "max_return_%": round(max_return, 2),
        "min_return_%": round(min_return, 2),
    }

    return results


if __name__ == "__main__":
    print("Starting breakout analysis...")  # Added print statement

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    tickers = get_nasdaq_100_tickers()

    print(f"Analyzing {len(tickers)} stocks...")

    results = []
    for ticker in tqdm(tickers, desc="Analyzing stocks"):
        print(ticker)
        data = fetch_stock_data(
            ticker, start_date, end_date
        )  # ,volume_threshold=1.0, price_threshold=2, holding_period=10)
        if data is None:
            print(f"No data found for {ticker}")
            continue
        results.append(data)

    analysis = analyze_aggregate_breakouts(results)

    print("\nBreakout Analysis Results:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
