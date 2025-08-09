import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import date
import math
import itertools
import os


def price_contract(in_dates, in_prices, out_dates, out_prices, rate, storage_cost_rate, total_vol, injection_withdrawal_cost_rate):
    volume = 0
    buy_cost = 0
    cash_in = 0
    all_dates = sorted(set(in_dates + out_dates))

    for start_date in all_dates:
        if start_date in in_dates:
            if volume <= total_vol - rate:
                volume += rate
                buy_cost += rate * in_prices[in_dates.index(start_date)]
                buy_cost += rate * injection_withdrawal_cost_rate
            else:
                return None  
        elif start_date in out_dates:
            if volume >= rate:
                volume -= rate
                cash_in += rate * out_prices[out_dates.index(start_date)]
                cash_in -= rate * injection_withdrawal_cost_rate
            else:
                return None 

    store_cost = math.ceil((max(out_dates) - min(in_dates)).days // 30) * storage_cost_rate
    return cash_in - store_cost - buy_cost


@st.cache_data
def load_data():
    file_path = 'Nat_Gas.csv'
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, file_path)
    df = pd.read_csv(full_path, parse_dates=['Dates'])
    return df

df = load_data()
prices = df['Prices'].values
dates = df['Dates'].values


start_date = pd.Timestamp('2020-10-31')

days_from_start = [(pd.Timestamp(d) - start_date).days for d in dates]
time = np.array(days_from_start)
sin_time = np.sin(time * 2 * np.pi / 365)
cos_time = np.cos(time * 2 * np.pi / 365)
X = np.column_stack((time, sin_time, cos_time))
y = prices

rf_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
rf_model.fit(X, y)
y_pred = rf_model.predict(X)
mse = mean_squared_error(y, y_pred)
st.write(f"**MSE**: {mse:.4f}")


def interpolate_ml(date):
    days = (pd.Timestamp(date) - start_date).days
    sin_component = np.sin(days * 2 * np.pi / 365)
    cos_component = np.cos(days * 2 * np.pi / 365)
    return rf_model.predict([[days, sin_component, cos_component]])[0]


st.subheader(" Predicting Natural Gas Prices with Machine Learning")


injection_dates = st.multiselect(
    "Choose injection date (s) Before 2030", 
    pd.date_range(start=start_date, periods=365*5).date,
    default=[start_date.date()]
)
withdrawal_dates = st.multiselect(
    "Choose withdrawal date (s) Before 2030", 
    pd.date_range(start=start_date, periods=365*5).date ,
    default=[start_date.date()]
)

rate = st.number_input("Injection/withdrawal rate (cf)", value=1000000)
storage_cost_rate = st.number_input("Storage Cost ($)", value=10000)
max_storage_volume = st.number_input("Max Storage (cf)", value=50000000)
injection_withdrawal_cost_rate = st.number_input("Cost of injection or withdrawal ($/cf)", value=0.0005)

if st.button("All possible contracts"):
    injection_dates = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates) 
    if not injection_dates or not withdrawal_dates:
        st.warning("At least input 1 injection and withdrawal date")
    else:
        results = []
        predicted_prices = []
        predicted_dates = []
        preded = []


        for i in range(len(injection_dates)):
            for j in range(len(withdrawal_dates)):
                if pd.Timestamp(withdrawal_dates[j]) > pd.Timestamp(injection_dates[i]):
                    in_price = interpolate_ml(injection_dates[i])
                    out_price = interpolate_ml(withdrawal_dates[j])
                    value = price_contract(
                        in_dates=[pd.Timestamp(injection_dates[i])],
                        in_prices=[in_price],
                        out_dates=[pd.Timestamp(withdrawal_dates[j])],
                        out_prices=[out_price],
                        rate=rate,
                        storage_cost_rate=storage_cost_rate,
                        total_vol=max_storage_volume,
                        injection_withdrawal_cost_rate=injection_withdrawal_cost_rate
                    )
                    if value is not None:
                        results.append({
                            "Injection Date": injection_dates[i],
                            "Injection Price": round(in_price, 2),
                            "Withdrawal Date": withdrawal_dates[j],
                            "Withdrawal Price": round(out_price, 2),
                            "Contract Value": round(value, 2)
                        })
                    if injection_dates[i] in preded :
                        continue
                    else:
                        preded.append(injection_dates[i])
                        predicted_prices.append(in_price)
                        
                        predicted_dates.append(pd.Timestamp(injection_dates[i]))
                        
                    if  withdrawal_dates[j] in preded:
                        continue
                    else:
                        preded.append(withdrawal_dates[j])
                        predicted_dates.append(pd.Timestamp(withdrawal_dates[j]))
                        predicted_prices.append(out_price)
            
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            
            best_contract = df_results.loc[df_results['Contract Value'].idxmax()]
            st.success(f"Highest price contract: {best_contract.to_dict()}")
            st.subheader("Predicted Prices Visualization (It takes a while to load)")
            fig, ax = plt.subplots()
            

            ax.scatter(sorted(predicted_dates), predicted_prices, label='Price Prediction')
            ax.plot(dates, prices, label='Actual Prices', markersize=2, alpha=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ')
            ax.set_title('Natural Gas Price Prediction Over Time')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
        else:
            st.error("Illegal contract dates or no profitable contracts found.")

            


