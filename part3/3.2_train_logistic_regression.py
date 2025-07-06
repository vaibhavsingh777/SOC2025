import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data with binary target
df = pd.read_csv('data_with_binary_target.csv')

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# Split data into train and test sets (80% train, 20% test), no shuffle to preserve time order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict on test set
y_pred = log_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy on test set: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

# Save results for plotting
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv('logistic_regression_results.csv', index=False)
print("Results saved to 'logistic_regression_results.csv'.")
