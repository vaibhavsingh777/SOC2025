import yfinance as yf
import pandas as pd
import datetime
import os

# List of NSE symbols
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

# Set period to 1 year
period = "1y"

all_data = []

for symbol in symbols:
    print(f"Fetching 1-year daily data for {symbol}...")
    try:
        data = yf.download(symbol, period=period, interval="1d")
        if data.empty:
            print(f"No data for {symbol}, skipping.")
            continue
        data.reset_index(inplace=True)
        data['Symbol'] = symbol
        # Optionally, add basic info (commented out for speed)
        # info = yf.Ticker(symbol).info
        # data['MarketCap'] = info.get('marketCap')
        # data['PE'] = info.get('trailingPE')
        # data['PB'] = info.get('priceToBook')
        # data['Sector'] = info.get('sector')
        all_data.append(data)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if all_data:
    df = pd.concat(all_data, ignore_index=True)
    # Keep only essential columns
    columns_to_keep = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[columns_to_keep]
    # Save to CSV
    csv_filename = 'nse_top40_1year_daily.csv'
    df.to_csv(csv_filename, index=False)
    # Check file size
    size_mb = os.path.getsize(csv_filename) / (1024 * 1024)
    print(f"CSV file size: {size_mb:.2f} MB")
else:
    print("No data was fetched for any symbol.")
