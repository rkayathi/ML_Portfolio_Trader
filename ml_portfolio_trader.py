import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MLPortfolioTrader:
    def __init__(self, tickers, price_data, market_data, confidence_threshold=0.6, max_drawdown=0.2):
        self.tickers = tickers
        self.raw_data = price_data
        self.market_returns = market_data["Adj Close"].pct_change()
        self.confidence_threshold = confidence_threshold
        self.max_drawdown = max_drawdown
        self.featured_data = {}
        self.models = {}
        self.scalers = {}
        self.predicted_probs = {}
        self.test_returns = {}
        self.optimal_weights = {}

    def calculate_features(self, df):
        df = df.copy()
        df["Return_1d"] = df["Adj Close"].pct_change()
        df["SMA_5"] = df["Adj Close"].rolling(5).mean()
        df["SMA_10"] = df["Adj Close"].rolling(10).mean()
        df["SMA_20"] = df["Adj Close"].rolling(20).mean()
        df["Volatility_5"] = df["Return_1d"].rolling(5).std()
        df["Volatility_10"] = df["Return_1d"].rolling(10).std()
        df["Momentum_5"] = df["Adj Close"].pct_change(periods=5)
        df["Momentum_10"] = df["Adj Close"].pct_change(periods=10)
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Price_vs_MA20"] = df["Adj Close"] / df["SMA_20"]
        df["Zscore_Price"] = (df["Adj Close"] - df["SMA_20"]) / df["Adj Close"].rolling(20).std()

        df = df.merge(self.market_returns.rename("Market_Return"), left_on="Date", right_index=True)
        df["Beta_30d"] = df["Return_1d"].rolling(30).cov(df["Market_Return"]) / df["Market_Return"].rolling(30).var()

        df["Target"] = (df["Adj Close"].shift(-1) > df["Adj Close"]).astype(int)
        return df.dropna()

    def prepare_all_data(self):
        for ticker in self.tickers:
            df = self.raw_data[self.raw_data["Ticker"] == ticker]
            self.featured_data[ticker] = self.calculate_features(df)

    def train_models(self):
        for ticker, df in self.featured_data.items():
            features = [
                "Return_1d", "SMA_5", "SMA_10", "SMA_20", "Volatility_5", "Volatility_10",
                "Momentum_5", "Momentum_10", "Volume_Change", "Price_vs_MA20",
                "Zscore_Price", "Beta_30d"
            ]
            X = df[features]
            y = df["Target"]
            future_return = df["Return_1d"].shift(-1)

            X_train, X_test, y_train, y_test, ret_train, ret_test = train_test_split(
                X, y, future_return, test_size=0.2, shuffle=False
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train_scaled, y_train)

            self.models[ticker] = model
            self.scalers[ticker] = scaler
            self.predicted_probs[ticker] = model.predict_proba(X_test_scaled)[:, 1]
            self.test_returns[ticker] = ret_test.reset_index(drop=True)

    def optimize_portfolio(self):
        confidences = pd.DataFrame(self.predicted_probs)
        returns = pd.DataFrame(self.test_returns)
        mask = confidences > self.confidence_threshold
        masked_returns = returns * mask

        expected_returns = masked_returns.mean()
        cov_matrix = masked_returns.cov()

        def neg_sharpe(weights, mu, cov):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return -port_return / port_vol if port_vol != 0 else 1e6

        def max_drawdown_penalty(weights, returns_df):
            portfolio_returns = returns_df.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (peak - cumulative) / peak
            return drawdown.max() - self.max_drawdown

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: self.max_drawdown - max_drawdown_penalty(x, masked_returns)}
        ]
        bounds = [(0, 1)] * len(self.tickers)
        initial_guess = [1 / len(self.tickers)] * len(self.tickers)

        opt_result = minimize(neg_sharpe, initial_guess, args=(expected_returns.values, cov_matrix.values),
                              method='SLSQP', bounds=bounds, constraints=constraints)

        self.optimal_weights = dict(zip(self.tickers, opt_result.x))
        return pd.DataFrame.from_dict(self.optimal_weights, orient='index', columns=["Optimal Weight"])

    def backtest(self):
        returns = pd.DataFrame(self.test_returns)
        confidences = pd.DataFrame(self.predicted_probs)
        mask = confidences > self.confidence_threshold
        masked_returns = returns * mask

        weights = np.array(list(self.optimal_weights.values()))
        portfolio_returns = masked_returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_returns, label='Portfolio Value')
        plt.title('Backtest: Cumulative Portfolio Returns')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return cumulative_returns


def get_data(tickers, start="2018-01-01", end="2024-12-31"):
    stock_data = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df["Ticker"] = ticker
        df = df.reset_index()[["Date", "Adj Close", "Volume", "Ticker"]]
        stock_data.append(df)
    return pd.concat(stock_data).reset_index(drop=True)

def get_sp500_data(start="2018-01-01", end="2024-12-31"):
    sp500 = yf.download("^GSPC", start=start, end=end)
    return sp500[["Adj Close"]]

def main():
    tickers = ["AAPL", "TSLA", "NVDA"]
    price_data = get_data(tickers)
    sp500_data = get_sp500_data()

    trader = MLPortfolioTrader(tickers, price_data, sp500_data, confidence_threshold=0.6, max_drawdown=0.2)
    trader.prepare_all_data()
    trader.train_models()
    weights_df = trader.optimize_portfolio()
    print(weights_df)
    trader.backtest()

if __name__ == "__main__":
    main()
