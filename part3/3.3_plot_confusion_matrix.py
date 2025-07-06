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
