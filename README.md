# TradeSeries

TradeSeries is a comprehensive time series analysis and forecasting project that focuses on predicting stock prices, specifically using the S&P 500 index as a baseline. The project leverages statistical methods and machine learning models to analyze historical market data, forecast future prices, and estimate volatility.

## Features

- **Data Fetching & Preprocessing**: Automatically downloads historical stock data using `yfinance`.
- **Exploratory Data Analysis (EDA)**: Visualizes closing prices alongside technical indicators like Bollinger Bands and MACD. Performs Stationarity Tests (ADF).
- **Interactive Web Dashboard**: Includes a user-friendly Streamlit web application to easily toggle between models and view plots interactively.
- **Advanced Modeling Techniques**:
  - **ARIMA**: AutoRegressive Integrated Moving Average for univariate time series forecasting.
  - **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity for volatility modeling and forecasting.
  - **Prophet**: Developed by Meta, robust to missing data and shifts in the trend.
  - **K-NN (K-Nearest Neighbors)**: Non-parametric method used for regression in time series.
  - **NNETAR / LSTM**: Advanced neural network architecture leveraging Long Short-Term Memory models to capture non-linear patterns.

## Directory Structure

- `Scripts/`: Contains all the Python scripts for models, data visualization, and the Streamlit app.
  - `main.py`: Centralized script to run all models sequentially.
  - `streamlit_app.py`: Streamlit application for the interactive dashboard.
  - `arima_model.py`, `garch_model.py`, `prophet_model.py`, `knn_model.py`, `nnetar_model.py`: Individual scripts housing the logic for respective models.
  - `data_visualization.py`: Functions for visualizing the raw dataset and performing technical analysis.
  - `requirements.txt`: List of required Python packages.
- `Images/`: Directory designated for storing output plots or documentation images.
- `Equations/`: Directory for storing mathematical formulas or markdown equations used for documentation.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/lav24fr/TradeSeries
   cd TradeSeries
   ```

2. **Set up a virtual environment (Optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r Scripts/requirements.txt
   ```

## Usage

### Run the Interactive Dashboard

The easiest way to interact with the project is via the Streamlit web application:

```bash
streamlit run Scripts/streamlit_app.py
```

### Run the Analysis Sequentially

If you prefer to run all models and analyses sequentially without the web interface:

```bash
python Scripts/main.py
```

## Technologies Used

- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical computations.
- **Statsmodels & Arch**: Statistical models (ARIMA, GARCH).
- **Prophet**: Time series forecasting.
- **Scikit-Learn**: Machine learning utilities and K-NN.
- **TensorFlow**: Deep learning framework for LSTM/NNETAR models.
- **Matplotlib & mplfinance**: Data visualization.
- **Streamlit**: Web dashboard framework.
