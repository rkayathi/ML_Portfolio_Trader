📈 MLPortfolioTrader
MLPortfolioTrader is a Python-based quantitative trading framework that uses machine learning (XGBoost) to predict short-term momentum and dynamically optimize a portfolio of stocks based on confidence thresholds and risk constraints such as maximum drawdown.

🚀 Features
Machine Learning Prediction: Predicts next-day price movement using XGBoost and a rich feature set including momentum, volatility, and beta.

Feature Engineering: Computes 12+ features per stock including moving averages, volume shifts, and market-relative metrics.

Risk-Constrained Portfolio Optimization: Allocates capital using a Sharpe ratio optimizer constrained by max drawdown and confidence thresholds.

Backtesting: Simulates returns vs. a benchmark over time and visualizes portfolio value growth.

Modular Design: Easy to adapt for different tickers, timeframes, or classification models.

🧠 Model Architecture
Target: Predicts whether the next day's return is positive.

Model: XGBClassifier

Thresholding: Trades only when predicted confidence exceeds a user-defined threshold (e.g. 0.6).

Portfolio Construction: Optimizes for maximum Sharpe ratio while maintaining a max drawdown ceiling (default: 20%).

📊 Key Metrics Computed
Sharpe Ratio (used in optimization)

Maximum Drawdown

Beta (30-day rolling) per asset

Cumulative Returns (plotted)

Dynamic Portfolio Allocation based on model confidence

📦 Installation & Requirements
bash
Copy
Edit
pip install yfinance xgboost scikit-learn matplotlib pandas numpy scipy
📁 File Structure
MLPortfolioTrader: Main class with all trading logic.

main(): Loads price data, trains models, allocates portfolio, and runs backtest.

get_data(): Downloads historical prices for a list of tickers.

get_sp500_data(): Downloads S&P 500 index data (used for beta calculation).

🛠 How It Works
Download Data
Historical stock and S&P 500 data from Yahoo Finance.

Feature Engineering
Technical indicators + market-relative metrics.

Train XGBoost Models
One model per ticker; trained to classify next-day price direction.

Predict & Threshold
Predict probabilities of upward movement and filter by confidence.

Portfolio Optimization
Allocates capital using a constrained Sharpe ratio optimizer.

Backtest
Applies trading logic to test data and plots cumulative returns.

