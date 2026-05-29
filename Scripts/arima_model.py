import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def run_arima_model(df, return_results=False):
    """Fits an ARIMA model, performs diagnostics, and forecasts."""
    print("--- Running ARIMA Model with statsmodels ---")
    close_prices = df['Close']
    
    # Define and fit the ARIMA model
    # The (5,1,2) order is based on the analysis from the original R script.
    model = ARIMA(close_prices, order=(5, 1, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    # Plotting fitted values vs actual data
    fitted_values = model_fit.fittedvalues
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(close_prices.index, close_prices, label='Actual Data')
    plt.plot(fitted_values.index, fitted_values, color='red', label='ARIMA Fitted')
    plt.title('ARIMA Fitted vs. Actual Data')
    plt.legend()
    if not return_results:
        plt.show()

    # Ljung-Box Test on Residuals
    residuals = model_fit.resid
    lb_test = acorr_ljungbox(residuals, lags=20)
    print(lb_test)

    # Forecasting for the next 30 days
    n_periods = 30
    forecast_result = model_fit.forecast(steps=n_periods)
    
    # Append the last historical point to connect the lines visually
    plot_forecast = pd.concat([close_prices.iloc[-1:], forecast_result])
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(close_prices.index, close_prices, label='Actual Data')
    plt.plot(plot_forecast.index, plot_forecast, color='red', label='Forecast')
    plt.title('ARIMA 30-Day Forecast')
    plt.legend()
    if not return_results:
        plt.show()

    if return_results:
        return model_fit, lb_test, fig1, fig2

if __name__ == '__main__':
    # For standalone execution, fetch data
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04')
    run_arima_model(df)