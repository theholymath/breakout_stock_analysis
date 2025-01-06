# app.py

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import uuid
import numpy as np
from io import StringIO
from openai import OpenAI

# Set the page configuration
st.set_page_config(
    page_title="📈 Stock Breakout Report Generator",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 Stock Breakout Report Generator")

# Sidebar for user inputs
st.sidebar.header("Enter Stock Parameters")


def get_default_dates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date


# User Inputs
ticker = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)", value="AAPL").upper()

# Updated: Separate Date Inputs
start_date = st.sidebar.date_input(
    "Start Date",
    value=get_default_dates()[0],
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now(),
)

end_date = st.sidebar.date_input(
    "End Date",
    value=get_default_dates()[1],
    min_value=start_date,
    max_value=datetime.now(),
)

volume_threshold = st.sidebar.number_input(
    "Volume Threshold (%)",
    min_value=0.0,
    max_value=10000.0,
    value=200.0,
    step=10.0,
    help="Percentage above the 20-day average volume to qualify as a breakout.",
)

price_threshold = st.sidebar.number_input(
    "Price Change Threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=2.0,
    step=0.5,
    help="Percentage increase in closing price to qualify as a breakout.",
)

holding_period = st.sidebar.number_input(
    "Holding Period (Days)",
    min_value=1,
    max_value=365,
    value=10,
    step=1,
    help="Number of days to hold the stock after a breakout.",
)

# New: Optional "Use AI" Feature
use_ai = st.sidebar.checkbox("🔍 Use AI for Summaries and Insights", value=False)

# Initialize OpenAI API Key (if AI is used)
if use_ai:
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Provide your OpenAI API key to enable AI-generated summaries and insights.",
    )
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
    )
    if not openai_api_key:
        st.sidebar.warning("Please enter your OpenAI API key to use AI features.")
else:
    openai_api_key = None


# Function to generate summary using OpenAI ChatCompletion API
def generate_summary(report_df, ticker, api_key):
    if api_key is None:
        return "AI Summary not enabled."

    num_breakouts = report_df["Breakout"].sum()
    average_return = report_df["Return (%)"].mean()
    cumulative_return = report_df["Return (%)"].sum()

    summary_prompt = f"""
    Analyze the following breakout report for {ticker}:

    - Total Breakout Days: {num_breakouts}
    - Average Return per Trade: {average_return:.2f}%
    - Cumulative Return: {cumulative_return:.2f}%

    Provide a concise summary highlighting the key performance metrics and any notable observations.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use "gpt-4" if available
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in financial analysis.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# Generate Report Button
if st.sidebar.button("Generate Report"):
    if ticker == "":
        st.error("Please enter a valid ticker symbol.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    elif use_ai and not openai_api_key:
        st.error("Please enter your OpenAI API key to use AI features.")
    else:
        try:
            with st.spinner("Fetching and processing data..."):
                # Fetch historical data using yfinance.Ticker
                ticker_obj = yf.Ticker(ticker)
                # Adding buffer for rolling calculations
                buffer_days = 60
                fetch_start = start_date - timedelta(days=buffer_days)
                raw_data = ticker_obj.history(
                    start=fetch_start, end=end_date + timedelta(days=1)
                )  # inclusive end date

                if raw_data.empty:
                    st.error("No data found for the given ticker and date range.")
                else:
                    # Ensure 'Date' is a column and sort
                    data = raw_data.reset_index()
                    data["Date"] = pd.to_datetime(data["Date"])
                    data.sort_values("Date", inplace=True)
                    data.reset_index(drop=True, inplace=True)

                    # Convert 'Date' to naive datetime (remove timezone if any)
                    if data["Date"].dt.tz is not None:
                        data["Date"] = data["Date"].dt.tz_convert(None)

                    # Select required columns
                    required_columns = ["Date", "Close", "Volume"]
                    if not all(col in data.columns for col in required_columns):
                        st.error(
                            f"Missing required columns in data. Available columns: {data.columns.tolist()}"
                        )
                    else:
                        data = data[required_columns]

                        # Calculate 20-day average volume
                        data["Avg_Volume_20"] = (
                            data["Volume"]
                            .astype(float)
                            .rolling(window=20, min_periods=20)
                            .mean()
                            .round(2)
                        )

                        # Drop rows with NaN Avg_Volume_20
                        data.dropna(subset=["Avg_Volume_20"], inplace=True)
                        data.reset_index(drop=True, inplace=True)

                        # Calculate Volume Percentage and Price Change Percentage
                        data["Volume_Pct"] = (
                            data["Volume"] / data["Avg_Volume_20"]
                        ) * 100
                        # Replace inplace=True with assignment to avoid FutureWarning
                        data["Price_Change_Pct"] = data["Close"].pct_change() * 100
                        data["Price_Change_Pct"] = data["Price_Change_Pct"].fillna(
                            0
                        )  # Replace NaN with 0 for the first row

                        # Create Breakout Indicator
                        data["Breakout"] = np.where(
                            (data["Volume_Pct"] >= volume_threshold)
                            & (data["Price_Change_Pct"] >= price_threshold),
                            1,
                            0,
                        )

                        # Identify Breakout Days within the specified date range
                        breakout_days = data[
                            (data["Breakout"] == 1)
                            & (data["Date"] >= pd.to_datetime(start_date))
                            & (data["Date"] <= pd.to_datetime(end_date))
                        ]

                        # Initialize Buy/Sell columns
                        data["Buy Date"] = np.nan
                        data["Buy Price"] = np.nan
                        data["Sell Date"] = np.nan
                        data["Sell Price"] = np.nan
                        data["Return (%)"] = np.nan

                        for _, row in breakout_days.iterrows():
                            buy_date = row["Date"]
                            buy_price = row["Close"]
                            target_sell_date = buy_date + timedelta(days=holding_period)

                            # Find the first trading day on or after the target_sell_date
                            sell_data = data[data["Date"] >= target_sell_date]
                            if sell_data.empty:
                                continue

                            sell_row = sell_data.iloc[0]
                            sell_date = sell_row["Date"]
                            sell_price = sell_row["Close"]
                            return_pct = ((sell_price - buy_price) / buy_price) * 100

                            # Update Buy/Sell signals
                            mask = data["Date"] == buy_date
                            data.loc[mask, "Buy Date"] = buy_date.strftime("%Y-%m-%d")
                            data.loc[mask, "Buy Price"] = round(buy_price, 2)
                            data.loc[mask, "Sell Date"] = sell_date.strftime("%Y-%m-%d")
                            data.loc[mask, "Sell Price"] = round(sell_price, 2)
                            data.loc[mask, "Return (%)"] = round(return_pct, 2)

                        # Select relevant columns for the report
                        report_columns = [
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

                        for col in report_columns:
                            if col not in data.columns:
                                data[col] = np.nan  # Ensure all columns are present

                        report_df = data[report_columns]

                        # Generate unique report ID
                        report_id = uuid.uuid4().hex
                        report_filename = f"report_{report_id}.csv"

                        # Convert DataFrame to CSV in memory
                        csv_buffer = StringIO()
                        report_df.to_csv(csv_buffer, index=False)
                        csv_content = csv_buffer.getvalue()  # Get the string content

                        # Provide download link
                        st.success("Report generated successfully!")

                        st.download_button(
                            label="📥 Download CSV Report",
                            data=csv_content,  # Pass the string content
                            file_name=report_filename,
                            mime="text/csv",
                        )

                        # Display the report in the app
                        st.subheader("📊 Breakout Report")
                        st.dataframe(report_df)

                        # Generate Summary using OpenAI (if AI is enabled)
                        if use_ai and openai_api_key:
                            with st.spinner("Generating AI summary..."):
                                summary = generate_summary(
                                    report_df, ticker, openai_api_key
                                )
                                st.subheader("📝 Report Summary")
                                st.write(summary)
                        elif use_ai and not openai_api_key:
                            st.warning(
                                "AI summary is enabled, but no API key was provided."
                            )
                        else:
                            # Optionally, provide a static summary or skip
                            pass

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
