import yfinance as yf
import pandas as pd
import datetime
import os

# Top 40 trending NSE stocks (from your prior list)
symbols = [
    "RELIANCE.NS", "TATAMOTORS.NS", "ZYDUSLIFE.NS", "GESHIP.NS", "BAJAJFINSV.NS",
    "IREDA.NS", "ASHOKA.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
    "ADANIGREEN.NS", "JSWENERGY.NS", "ICICIGI.NS", "SHRIRAMFIN.NS", "LUPIN.NS",
    "AMBUJACEM.NS", "MARICO.NS", "KOTAKBANK.NS", "GODREJCP.NS", "SHREECEM.NS",
    "CONCOR.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "ITC.NS", "HINDUNILVR.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS",
    "BHARTIARTL.NS", "HCLTECH.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS",
    "BAJFINANCE.NS", "NESTLEIND.NS", "M&M.NS", "BRITANNIA.NS", "EICHERMOT.NS"
]

start_date = "2019-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

all_data = []

for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)
    if hist.empty:
        continue
    hist.reset_index(inplace=True)
    hist['Symbol'] = symbol
    info = ticker.info
    hist['MarketCap'] = info.get('marketCap')
    hist['PE'] = info.get('trailingPE')
    hist['PB'] = info.get('priceToBook')
    hist['Sector'] = info.get('sector')
    all_data.append(hist)

df = pd.concat(all_data, ignore_index=True)

# Keep only essential columns
df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap', 'PE', 'PB', 'Sector']]

# Save to CSV
df.to_csv('nse_top40_trending_2019_2025.csv', index=False)

# Check file size
size_mb = os.path.getsize('nse_top40_trending_2019_2025.csv') / (1024 * 1024)
print(f"CSV file size: {size_mb:.2f} MB")
