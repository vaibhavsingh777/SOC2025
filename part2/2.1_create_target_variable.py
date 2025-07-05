import pandas as pd

# Load the data from your CSV (generated in Part 1)
df = pd.read_csv('../part1/nse_top40_1year_daily.csv')

# Ensure data is sorted by Symbol and Date for correct shifting
df = df.sort_values(['Symbol', 'Date'])

# Create the target variable: next day's closing price for each stock
df['Next_Close'] = df.groupby('Symbol')['Close'].shift(-1)

# Drop rows where Next_Close is NaN (i.e., last day for each stock)
df = df.dropna(subset=['Next_Close'])

# Keep only the essential columns
columns_to_keep = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Close']
df = df[columns_to_keep]

# Save the cleaned DataFrame for the next steps
df.to_csv('data_with_target.csv', index=False)

print("Target variable 'Next_Close' created and saved in 'data_with_target.csv'.")
