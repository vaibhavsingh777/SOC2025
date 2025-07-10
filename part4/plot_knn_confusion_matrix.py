import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Choose K value for visualization (e.g., 5)
K = 5
results = pd.read_csv(f'knn_results_k{K}.csv')

# Compute confusion matrix
cm = confusion_matrix(results['Actual'], results['Predicted'])

# Plot confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'KNN (K={K}) – Confusion Matrix')
plt.show()
