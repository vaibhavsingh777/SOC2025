import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data with target variable
data = pd.read_csv('data_with_target.csv')

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data["Next_Close"]

# Split data into train and test sets (80% train, 20% test), no shuffle to preserve time order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")

# Save predictions and true values for plotting in next step
results = X_test.copy()
results['Actual_Next_Close'] = y_test.values
results['Predicted_Next_Close'] = y_pred
results.to_csv('linear_regression_results.csv', index=False)
print("Predictions saved to 'linear_regression_results.csv'.")
