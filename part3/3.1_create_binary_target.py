import pandas as pd

# Load the data with target variable from part2
df = pd.read_csv('../part2/data_with_target.csv')

# Create the binary target: 1 if next day's close > today's close, else 0
df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

# Save the updated DataFrame for classification
df.to_csv('data_with_binary_target.csv', index=False)

print("Binary target 'Target' created and saved in 'data_with_binary_target.csv'.")
