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

Setup
For this project, I used Python and installed libraries like yfinance, scikit-learn, pandas, matplotlib, and seaborn.

If you want to try this out, just make sure to install these packages before running the code.

Maybe later on, if I get time, I will include all these in a requirements.txt file separately.

Quick Install:

bash
pip install yfinance scikit-learn pandas matplotlib seaborn
Project Workflow
Part 1: Downloading Stock Data
Task: Fetch 1-Year Daily Data
Example:

Ticker: "AAPL" (You can also try "TCS.NS", "RELIANCE.NS", "TSLA", etc.)

Downloaded using yfinance and checked the data.

Part 2: Linear Regression – Predicting Next Day’s Close Price
Task 1: Create Target Variable

Made a new column for next day’s close price and cleaned the data.

Task 2: Train Linear Regression Model

Used features like Open, High, Low, Close, Volume.

Split data into train and test sets.

Trained the model and checked Mean Squared Error.

Task 3: Plot Actual vs Predicted

Plotted graphs to compare actual and predicted prices.

Part 3: Logistic Regression – Predicting Price Movement
Task 1: Define Binary Target

Created a column to indicate if price went up or down.

Task 2: Train Logistic Regression Model

Used the same features, split data, trained the model.

Checked accuracy and confusion matrix.

Part 4: K-Nearest Neighbors (KNN) Classification
Task: Evaluate KNN for Various K Values

Tried different values of k (like 3, 5, 7) and checked which one gives best accuracy.

Bonus Tasks (Optional)
Added technical indicators (like moving averages, RSI) and observed how accuracy changes.

Plotted confusion matrix heatmap using seaborn for better visualization.

Applied PCA before running KNN to reduce dimensions.

Compared all three models in a markdown summary.

Submission Checklist
Code for Linear Regression with performance plot

Code for Logistic Regression with confusion matrix

Code for KNN with varying k values

Answered bonus questions (optional)
