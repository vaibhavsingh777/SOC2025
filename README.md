Automated Trading Strategy Optimization Using Multi-Modal Reinforcement Learning

### Learning Objectives

Hello! In this project, I have explored some basic machine learning techniques(specifically using RL) for stock market data. Let me break down what I have done, part by part, in my own words.

1. Understanding and Implementing Linear Regression, Logistic Regression, and KNN

   ### Linear Regression

   For this project, I chose Linear Regression as my starting point because it is a well-established and interpretable method for predicting continuous values. I used today’s open, high, low, close, and volume as input features to estimate the next day’s closing price. The underlying assumption is that there is a linear relationship between these features and the target price, which can be represented as: $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$$

   Here, y is the predicted closing price, and $x_1$ to $x_n$ are the selected features. I selected these particular variables because, from my understanding and research, they capture the essential daily market dynamics. I trained the model using historical stock data, applying the Ordinary Least Squares (OLS) method to minimize prediction errors.

   One advantage I found with Linear Regression is its interpretability—the coefficients directly indicate how each feature influences the next day’s price. To evaluate the model, I used Mean Squared Error (MSE), which helped me measure the accuracy of my predictions. Overall, this approach gave me a strong baseline for stock price forecasting.

   ### Logistic Regression

   After working with regression, I moved to Logistic Regression, which is mainly used for classification tasks. In this project, I used Logistic Regression to predict whether the stock price will go up or down the next day. I transformed the problem into a binary classification by creating a target variable: if the next day’s closing price is higher than today’s, it is labeled as 1 (up); otherwise, it is 0 (down).

   Logistic Regression models the probability that a given input belongs to a particular class using the logistic (sigmoid) function. The mathematical form is:

   $$
   P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)}}
   $$

   Here, $$P(y=1|X)$$ is the probability that the price will go up, and the $$\beta$$ coefficients are learned from the data. I trained the model using the same features as before and evaluated its performance using accuracy and the confusion matrix, which shows how well the model distinguishes between up and down movements. Logistic Regression is particularly useful because it provides probabilistic outputs and is less sensitive to outliers compared to linear regression. This method helped me understand the likelihood of price movement direction and served as a solid approach for binary classification in financial data.

   ### K-Nearest Neighbors (KNN)

   Next, I explored the K-Nearest Neighbors (KNN) algorithm, which is a non-parametric and instance-based learning method. KNN classifies a new data point based on the majority class among its ‘k’ closest neighbors in the feature space. For this project, I used KNN to classify whether the stock price would go up or down the next day, using the same set of features.

   The main idea is simple: for each test instance, the algorithm calculates the distance (usually Euclidean) to all training points, finds the ‘k’ nearest ones, and assigns the class that is most common among them. I experimented with different values of $$k$$ (like 3, 5, and 7) to see which provided the best classification accuracy. KNN is intuitive and does not make strong assumptions about the underlying data distribution, making it flexible for various types of datasets.

   One thing I observed is that KNN’s performance can be sensitive to the choice of $$k$$ and the scale of features, so I made sure to preprocess the data appropriately. I evaluated the model using accuracy and confusion matrices, which helped me compare its effectiveness with Logistic Regression. Overall, KNN gave me practical insights into how neighborhood-based classification works for stock price movement prediction.

### Fetching and Processing Real Stock Data Using yfinance

I didn’t want to use any fake or sample data, so I used the yfinance library to download real stock data from Yahoo Finance. I tried with different tickers like AAPL, TCS.NS, RELIANCE.NS, and TSLA.

I fetched one year of daily data for each stock. After downloading, I checked the data, removed any missing values, and made sure it was clean and ready for analysis.

This step helped me understand how to work with real-world data, which is usually messy and needs some cleaning before using in any model.

3. Feature Engineering and Machine Learning for Prediction and Classification
   I created new features from the existing data to make the models work better. For example, I made a new column for the next day’s closing price (for regression) and a binary target column to indicate if the price went up or down (for classification).

I split the data into training and testing sets, so that I can train the models on one part and test on another to see how well they perform.

After training the models, I checked their performance using metrics like Mean Squared Error (for regression) and Accuracy/Confusion Matrix (for classification).

I also plotted graphs to visualize the actual vs predicted values, and confusion matrices to see where the models are making mistakes. This helped me understand the strengths and weaknesses of each approach.

# Setup

pip install yfinance scikit-learn pandas matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Part 1: Downloading Stock Data

Task 1.1: Fetch 1-Year Daily Data
ticker = "AAPL" # You can also try "TCS.NS", "RELIANCE.NS", "TSLA", etc.
data = yf.download(ticker, period="1y")
data.head()

# Part 2: Linear Regression – Predicting Next Day’s Close Price

Task 2.1: Create Target Variable
data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)
Task 2.2: Train Linear Regression Model
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data["Next_Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
Task 2.3: Plot Actual vs Predicted
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Linear Regression - Next Day Close Price')
plt.show()

# Part 3: Logistic Regression – Predicting Price Movement

Task 3.1: Define Binary Target
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)
Task 3.2: Train Logistic Regression Model
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#Part 4: K-Nearest Neighbors (KNN) Classification
Task 4.1: Evaluate KNN for Various K Values
python
Copy
Edit
for k in [3, 5, 7]:
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred_knn)
print(f"K={k}, Accuracy={acc}")
#Bonus Tasks
Add technical indicators (e.g., moving averages, RSI) and observe accuracy change

Plot confusion matrix heatmap using seaborn.heatmap

Apply PCA before running KNN to reduce dimensions

#Compare all three models in a markdown summary

1.  Submission Checklist
2.  Code for Linear Regression with performance plot
3.  Code for Logistic Regression with confusion matrix
4.  Code for KNN with varying k values
5.  Answered bonus questions (optional)
