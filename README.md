Automated Trading Strategy Optimization Using Multi-Modal Reinforcement Learning

- Learning Objectives
  Hello! In this project, I have explored some basic machine learning techniques(specifically using RL) for stock market data. Let me break down what I have done, part by part, in my own words.

1. Understanding and Implementing Linear Regression, Logistic Regression, and KNN
   Linear Regression
   For this project, I chose Linear Regression as my starting point because it is a well-established and interpretable method for predicting continuous values. I used today’s open, high, low, close, and volume as input features to estimate the next day’s closing price. The underlying assumption is that there is a linear relationship between these features and the target price, which can be represented as: $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$$

Here,
y
y is the predicted closing price, and x1 to xn are the selected features. I selected these particular variables because, from my understanding and research, they capture the essential daily market dynamics. I trained the model using historical stock data, applying the Ordinary Least Squares (OLS) method to minimize prediction errors.

One advantage I found with Linear Regression is its interpretability—the coefficients directly indicate how each feature influences the next day’s price. To evaluate the model, I used Mean Squared Error (MSE), which helped me measure the accuracy of my predictions. Overall, this approach gave me a strong baseline for stock price forecasting.

Logistic Regression
After that, I moved to Logistic Regression. This one is mainly for classification problems. So, I used it to predict whether the stock price will go up or down the next day. I converted the problem into a simple yes/no (1/0) type, and trained the model to classify the movement.

K-Nearest Neighbors (KNN)
Then, I tried KNN, which is a very intuitive algorithm. It looks at the ‘k’ closest data points and decides the class based on majority. I used KNN to classify the price movement, and also experimented with different values of ‘k’ to see which one works best.

2. Fetching and Processing Real Stock Data Using yfinance
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
