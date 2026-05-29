import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

import data_visualization as dv
import arima_model as am
import garch_model as gm
import prophet_model as pm
import knn_model as km
import nnetar_model as nm

st.set_page_config(page_title="S&P 500 Forecasting", page_icon="📈", layout="wide")

st.title("📈 S&P 500 Time Series Forecasting")
st.markdown("Analyze and forecast the S&P 500 index using various statistical and machine learning models.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Model / View", ["Data Overview", "ARIMA", "GARCH", "Prophet", "K-NN", "NNETAR"])

@st.cache_data
def load_data():
    # Need auto_adjust=False for Prophet as it was used in prophet_model.py's __main__ block
    df = yf.download('^GSPC', start='2015-01-01', end='2020-06-04', auto_adjust=False)
    # Prophet requires 'Close' to be flattened, which was handled inside the function
    return df

df = load_data()

if page == "Data Overview":
    st.header("📊 Data Overview & Visualization")
    st.write("Visualizing the S&P 500 closing prices, Bollinger Bands, MACD, and Stationarity Tests.")
    with st.spinner("Fetching data and generating plots..."):
        df_full, fig1, fig2, fig3, adf_result = dv.run_data_visualization(return_results=True)
    st.pyplot(fig1)
    st.pyplot(fig2)
    
    st.subheader("Stationarity Test (ADF)")
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")
    st.pyplot(fig3)

elif page == "ARIMA":
    st.header("📉 ARIMA Model")
    with st.spinner("Fitting ARIMA Model..."):
        model_fit, lb_test, fig1, fig2 = am.run_arima_model(df, return_results=True)
    st.pyplot(fig1)
    
    st.subheader("Ljung-Box Test on Residuals")
    st.dataframe(lb_test)
    st.pyplot(fig2)
    
    with st.expander("View Model Summary"):
        st.text(model_fit.summary().as_text())

elif page == "GARCH":
    st.header("📉 GARCH Model (Volatility)")
    with st.spinner("Fitting GARCH Model..."):
        garch_fit, fig1, fig2 = gm.run_garch_model(df, return_results=True)
    st.pyplot(fig1)
    st.pyplot(fig2)
    
    with st.expander("View Model Summary"):
        st.text(garch_fit.summary().as_text())

elif page == "Prophet":
    st.header("🔮 Prophet Model")
    with st.spinner("Fitting Prophet Model..."):
        # The script renames columns if they have 'GSPC.'
        df_prophet = df.copy()
        df_prophet.columns = [col.replace('GSPC.', '') for col in df_prophet.columns]
        model, forecast, fig1, fig2 = pm.run_prophet_model(df_prophet, return_results=True)
    st.pyplot(fig1)
    st.pyplot(fig2)

elif page == "K-NN":
    st.header("🤖 K-Nearest Neighbors (K-NN)")
    with st.spinner("Fitting K-NN Model..."):
        knn_model, rmse, fig1 = km.run_knn_model(df, return_results=True)
    st.pyplot(fig1)
    st.success(f"**Rolling Origin RMSE:** {rmse:.4f}")

elif page == "NNETAR":
    st.header("🧠 NNETAR (LSTM Neural Network)")
    with st.spinner("Training LSTM Model (this may take a minute)..."):
        model, fig1 = nm.run_nnetar_model(df, return_results=True)
    st.pyplot(fig1)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit")
