import pandas as pd
import matplotlib.pyplot as plt

# Load the results file
results = pd.read_csv('linear_regression_results.csv')

# Plot actual vs predicted closing prices
plt.figure(figsize=(12,6))
plt.plot(results['Actual_Next_Close'].values, label='Actual')
plt.plot(results['Predicted_Next_Close'].values, label='Predicted')
plt.legend()
plt.title('Linear Regression - Next Day Close Price')
plt.xlabel('Test Data Points')
plt.ylabel('Price')
plt.show()
