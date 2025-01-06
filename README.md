# App: https://dubvax5c3s9dmrscgd9ggs.streamlit.app/
Go play with it! 

If you have an OpenAI API key you can get "AI Analsys" by checking the AI box in the sidebar. The AI analysis will be below the data. 

# Breakout Stocks
A small quick project inspired by a job application project

# Analysis
I wanted to see if there is any value in this strategy. I looked at the NasDaq 100 for a year and the average returns were negative. That is, if you invest when a stock "breaks out" for the Nasdq 100 over the last year you would have lost around 1%. you can run `python breakout_analysis.py` to see the output. As of Januray 6, 2025, the output is 

```bash
Breakout Analysis Results:
count_breakout_trades: 282
average_return_%: -0.47
median_return_%: -1.1
max_return_%: 38.55
min_return_%: -21.02
```

---

## Project Description
Mini Strategy Analysis Project

The goal of this mini project is to test your ability to setup a full mini project from beginning to scratch which involves getting the right data, using external APIs, looking out for lookahead bias, detail orientedness and usability.  We prioritize speed and accuracy at Quanta so please try to accomplish this task as quickly as possible, but with 100% accuracy. In finance, accuracy is critically important so ensure that there’s no lookahead bias or other data issues. 

Challenge: 

Write a python program to test this thesis: 
If a stock has volume on a day that is >200% (X%) the average daily volume over the last 20 days AND it is up at least 2% (Y) on that day compared to the previous day, it shows it might be “breaking out”. We should buy it when that breakout happens and hold it for 10 (Z) days and see what the return is if we did that every time. 

Your task is to create a program that works like this: 
User goes to a link and inputs the following: 
Ticker
Start Date and End Date
Percent Volume Breakout Threshold (Minimum Percent that the volume that day is above the last 20 days average volume so if set at 200% that means the volume that day is 200% greater than the average volume over the last 20 days) 
Daily Change Threshold (Minimum Percent that the stock is up on the day of the volume breakout over the previous day)
Holding Period (How many days to hold the investment before selling)
Generate Report Button
After clicking generate report, it should create a google sheet or downloadable CSV with statistics that shows what days exceeded the volume breakout threshold AND the price change threshold. Those are the “buy” days. Then it shows the results for each of those days which means what would be the return if the investor bought at the close of that breakout day and held for 10 (Z) days.  


Deliverables: 
Share the URL link to your program. It should be hosted at a weblink that we can click on and start inputting directly with a basic U.I. created. (There should be a bunch of free ways to host). 
Explain your process of how you set it up (what data you used etc) and any roadblocks you had in setting everything up and how much time the whole project took you
EXTRA CREDIT: Play with the program using different tickers and share your insights on whether this looks like a useful signal and what other parameters you would add next to make it better.

