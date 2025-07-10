# Part 4: K-Nearest Neighbors (KNN) Classification

In this section, I investigated the effectiveness of the **K-Nearest Neighbors (KNN)** algorithm for predicting next-day stock price movement. Building on the features and binary target from previous parts, I systematically evaluated KNN for different values of $$ K $$.

## 1. Data Preparation

I used the same features as before: `Open`, `High`, `Low`, `Close`, `Volume`.  
The target variable is binary:

- **1** if the next day's close is higher than today's
- **0** otherwise

import pandas as pd
from sklearn.model_selection import train_test_split

# Load data

df = pd.read_csv('data_with_binary_target.csv')
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# Split data (80% train, 20% test), preserving time order

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, shuffle=False
)

## 2. KNN Model Evaluation

I evaluated KNN for $$ K = 3, 5, 7 $$, recording the accuracy for each case and saving the predictions for further analysis.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for k in :
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"K={k}, Accuracy={acc:.4f}")

    # Save results for each K
    results = X_test.copy()
    results['Actual'] = y_test.values
    results['Predicted'] = y_pred
    results.to_csv(f'knn_results_k{k}.csv', index=False)

## 3. Visualizing the Confusion Matrix

To better understand model performance, I visualized the confusion matrix for any chosen $$ K $$:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Choose K (e.g., 5)

K = 5
results = pd.read_csv(f'knn_results_k{K}.csv')
cm = confusion_matrix(results['Actual'], results['Predicted'])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'KNN (K={K}) – Confusion Matrix')
plt.show()

## Key Points

- **KNN** uses proximity in feature space to classify price movement direction.
- **Accuracy** varies with $$ K $$; results are saved for each value for comparison.
- **Confusion matrix** visualization highlights where the model succeeds and fails.

## Files Generated

- `knn_results_k3.csv`, `knn_results_k5.csv`, `knn_results_k7.csv` — Predictions for each $$ K $$.
- (Requires `data_with_binary_target.csv` from Part 3.)

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

## How to Run

1. Ensure all dependencies are installed.
2. Run the code blocks in order.
3. Review accuracy for each $$ K $$.
4. Visualize confusion matrices for deeper insight.

Through this process, I gained practical understanding of how KNN's neighborhood-based approach performs in the context of financial time series classification.
