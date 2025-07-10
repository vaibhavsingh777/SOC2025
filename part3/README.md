# Part 3: Logistic Regression – Predicting Stock Price Movement

In this section, I set out to answer a fundamental question: **Can I predict whether a stock's closing price will rise or fall the next day, using only historical price and volume data?** To tackle this, I employed logistic regression—a classic tool for binary classification. Here, I detail my process, the rationale behind each step, and the insights (and limitations) I encountered.

## 1. Data Preparation

I began by ensuring my dataset was both clean and structured for the task. Drawing inspiration from my earlier work in Part 2, I made sure to:

- **Sort the data** by symbol and date to maintain chronological order for each stock.
- **Create a target variable**: For each trading day, I calculated the next day's closing price (`Next_Close`) for that stock. This required shifting the 'Close' column within each symbol group.
- **Filter out incomplete rows**: Any row where `Next_Close` was missing (i.e., the last day for a stock) was dropped.
- **Retain only essential columns** for clarity and efficiency.

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

## 2. Creating the Binary Target

To frame this as a classification problem, I needed a binary target:

- **1** if the next day's close was higher than today's,
- **0** otherwise.

This approach simplified the prediction task to a straightforward "up or down" movement.

import pandas as pd

# Load the data with target variable

df = pd.read_csv('data_with_target.csv')

# Create the binary target: 1 if next day's close > today's close, else 0

df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

# Save the updated DataFrame for classification

df.to_csv('data_with_binary_target.csv', index=False)

print("Binary target 'Target' created and saved in 'data_with_binary_target.csv'.")

## 3. Model Training and Evaluation

Armed with my features (`Open`, `High`, `Low`, `Close`, `Volume`) and the binary target, I proceeded to:

- **Split the data**: 80% for training, 20% for testing, with no shuffling to preserve the time series nature.
- **Train a logistic regression model**: I chose logistic regression for its interpretability and baseline performance.
- **Predict on the test set** and **evaluate** using accuracy and a confusion matrix.

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

## 4. Visualizing Model Performance

Numbers alone rarely tell the full story. To better understand where my model succeeded and failed, I visualized the confusion matrix as a heatmap. This made it clear how often the model correctly predicted "up" or "down" days—and where it faltered.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the results file

results = pd.read_csv('logistic_regression_results.csv')

# Compute confusion matrix

cm = confusion_matrix(results['Actual'], results['Predicted'])

# Plot confusion matrix heatmap

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression – Confusion Matrix')
plt.show()

## Reflections & Insights

- **What I achieved:** I successfully built a pipeline that predicts next-day price movement using only basic price and volume features. The workflow is reproducible and easy to extend.
- **What I learned:** Logistic regression provides a simple, interpretable baseline for this task. The confusion matrix reveals not just overall accuracy, but also the types of errors made.
- **What was missing:** While this approach is systematic, it does not capture more complex market dynamics, such as trends, volatility, or external factors. I realized that incorporating additional features or more sophisticated models could further improve prediction accuracy.

## Key Files

- `data_with_target.csv` – Cleaned data with next day's close.
- `data_with_binary_target.csv` – Data with the binary target variable.
- `logistic_regression_results.csv` – Test set with actual and predicted labels.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

## How to Reproduce

1. Install all dependencies.
2. Run each code block in order.
3. Examine the printed accuracy and confusion matrix.
4. Visualize the confusion matrix using the provided plotting code.

## Conclusion

Through this process, I learned that even simple models like logistic regression can offer valuable insights into stock price movement—though they have their limits. This project forms a solid baseline for more advanced experimentation in financial prediction.
