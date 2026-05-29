import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_model(df, return_results=False):
    """Fits a GARCH model and forecasts volatility."""
    print("--- Running GARCH Model ---")
    close_prices = df['Close']
    returns = 100 * close_prices.pct_change().dropna()

    # Fitting GARCH(1,1) with ARMA mean model
    garch_model = arch_model(returns, mean='AR', lags=3, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    print(garch_fit.summary())

    # Conditional Volatility Plot
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(garch_fit.conditional_volatility)
    plt.title('Conditional Volatility Plot')
    plt.ylabel('Sigma(t)')
    if not return_results:
        plt.show()

    # Forecasting returns and volatility
    forecasts = garch_fit.forecast(horizon=30, reindex=False)
    forecast_volatility = np.sqrt(forecasts.variance.values[-1, :])
    forecast_mean = forecasts.mean.values[-1, :]

    # 30 business days starting from the next day
    forecast_dates = pd.date_range(start=returns.index[-1], periods=31, freq='B')[1:]
    
    # Append the last historical point to connect the lines visually
    plot_dates = [returns.index[-1]] + list(forecast_dates)
    plot_mean = [returns.iloc[-1]] + list(forecast_mean)
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(returns.index, returns, label='Returns')
    plt.plot(plot_dates, plot_mean, color='red', label='Forecast Mean')
    
    # 95% Confidence Interval (1.96 * volatility)
    ci_lower = forecast_mean - 1.96 * forecast_volatility
    ci_upper = forecast_mean + 1.96 * forecast_volatility
    plt.fill_between(forecast_dates, ci_lower, ci_upper, color='gray', alpha=0.5, label='95% Confidence Interval')
    
    plt.title('GARCH 30-Day Forecast of Returns')
    plt.legend()
    if not return_results:
        plt.show()

    if return_results:
        return garch_fit, fig1, fig2

if __name__ == '__main__':
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    run_garch_model(df)