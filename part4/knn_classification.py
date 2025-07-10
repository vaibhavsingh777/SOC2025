import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data with binary target
df = pd.read_csv('data_with_binary_target.csv')

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# Split data into train and test sets (80% train, 20% test), no shuffle to preserve time order
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Evaluate KNN for various K values
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_knn)
    print(f"K={k}, Accuracy={acc:.4f}")

    # Save results for each K
    results = X_test.copy()
    results['Actual'] = y_test.values
    results['Predicted'] = y_pred_knn
    results.to_csv(f'knn_results_k{k}.csv', index=False)
    print(f"Results for K={k} saved to 'knn_results_k{k}.csv'.")
