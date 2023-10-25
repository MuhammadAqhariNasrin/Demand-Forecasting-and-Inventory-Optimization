# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:13:01 2023

@author: muham
"""

import streamlit as st
import subprocess


subprocess.call("pip install prophet", shell=True)
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Create a Streamlit app title
st.title("Demand Forecasting and Inventory Optimization")

# Create a Streamlit file uploader widget
uploaded_file = st.file_uploader("Upload your data file (CSV format)", type=["csv"])

# Check if a file has been uploaded by the user
if uploaded_file is not None:
    # Load the user-uploaded data
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Data preprocessing
    data = data.rename(columns={
        'Type': 'type',
        'Days for shipping (real)': 'days_for_shipping_real',
        'Days for shipment (scheduled)': 'days_for_shipment_scheduled',
        'Benefit per order': 'benefit_per_order',
        'Sales per customer': 'sales_per_customer',
        'Delivery Status': 'delivery_status',
        'Late_delivery_risk': 'late_delivery_risk',
        'Category Id': 'category_id',
        'Category Name': 'category_name',
        'Customer City': 'customer_city',
        'Customer Country': 'customer_country',
        'Customer Email': 'customer_email',
        'Customer Fname': 'customer_first_name',
        'Customer Id': 'customer_id',
        'Customer Lname': 'customer_last_name',
        'Customer Password': 'customer_password',
        'Customer Segment': 'customer_segment',
        'Customer State': 'customer_state',
        'Customer Street': 'customer_street',
        'Customer Zipcode': 'customer_zipcode',
        'Department Id': 'department_id',
        'Department Name': 'department_name',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Market': 'market',
        'Order City': 'order_city',
        'Order Country': 'order_country',
        'Order Customer Id': 'order_customer_id',
        'order date (DateOrders)': 'order_date_date_orders',
        'Order Id': 'order_id',
        'Order Item Cardprod Id': 'order_item_carprod_id',
        'Order Item Discount': 'order_item_discount',
        'Order Item Discount Rate': 'order_item_discount_rate',
        'Order Item Id': 'order_item_id',
        'Order Item Product Price': 'order_item_product_price',
        'Order Item Profit Ratio': 'order_item_profit_ratio',
        'Order Item Quantity': 'order_item_quantity',
        'Sales': 'sales',
        'Order Item Total': 'order_item_total',
        'Order Profit Per Order': 'order_profit_per_order',
        'Order Region': 'order_region',
        'Order State': 'order_state',
        'Order Status': 'order_status',
        'Order Zipcode': 'order_zipcode',
        'Product Card Id': 'product_card_id',
        'Product Category Id': 'product_category_id',
        'Product Description': 'product_description',
        'Product Image': 'product_image',
        'Product Name': 'product_name',
        'Product Price': 'product_price',
        'Product Status': 'product_status',
        'shipping date (DateOrders)': 'shipping_date_date_orders',
        'Shipping Mode': 'shipping_mode'
    })

    data.drop(columns=['product_description'], inplace=True)
    data['order_date_date_orders'] = pd.to_datetime(data['order_date_date_orders'], format='%m/%d/%Y %H:%M')
    data['shipping_date_date_orders'] = pd.to_datetime(data['shipping_date_date_orders'], format='%m/%d/%Y %H:%M')

    columns_to_convert = ['category_id', 'customer_id', 'customer_zipcode', 'department_id', 'order_customer_id', 'order_id', 'order_item_carprod_id', 'order_item_id', 'order_zipcode', 'product_card_id', 'product_category_id']

    data[columns_to_convert] = data[columns_to_convert].astype(str)

    data['order_year'] = pd.DatetimeIndex(data['order_date_date_orders']).year
    data['order_month'] = pd.DatetimeIndex(data['order_date_date_orders']).month
    data['order_week_day'] = pd.DatetimeIndex(data['order_date_date_orders']).day_name()

    data['order_date'] = pd.to_datetime(data['order_date_date_orders'])

    data['order_hour'] = pd.DatetimeIndex(data['order_date_date_orders']).hour
    data['order_month_year'] = pd.to_datetime(data['order_date_date_orders']).dt.to_period('M')
    data['year_week'] = data['order_date'].dt.to_period('W')

    weekly_orders = data.groupby('year_week')['order_item_quantity'].sum()

    weekly_orders_df = weekly_orders.reset_index()

    weekly_orders_df.columns = ['ds', 'y']

    weekly_orders_df['ds'] = weekly_orders_df['ds'].dt.to_timestamp()

    # Display information about the author and data source
    st.subheader("About the Author:")
    st.text("Name: Muhammad Aqhari Nasrin")
    st.text("Email: muhammad.aqhari.nasrin@gmail.com")
    st.markdown("LinkedIn: [Muhammad Aqhari Nasrin](https://www.linkedin.com/in/muhammad-aqhari-nasrin)")

    st.markdown("Data Source: [Data Set](https://data.mendeley.com/datasets/8gx2fvg2k6/5)")

    # Display a subheader with the original data
    st.subheader("Original Data:")
    st.write(data.head(50))
    st.write(data.tail(50))

    # Data preprocessing and analysis code...
    
    # Split point for training and testing data
    split_point = int(len(weekly_orders_df) * 0.80)
    train = weekly_orders_df.iloc[:split_point]
    test = weekly_orders_df.iloc[split_point:]

    # User input elements
    
    weekly_model = Prophet()
    weekly_model.fit(train)

    weekly_future = weekly_model.make_future_dataframe(periods=len(test), freq='W-SUN')
    weekly_forecast = weekly_model.predict(weekly_future)

    # Calculate and define the metrics
    y_pred_train = weekly_forecast['yhat'][:split_point]
    mae_train = mean_absolute_error(train['y'], y_pred_train)
    mse_train = mean_squared_error(train['y'], y_pred_train)
    rmse_train = np.sqrt(mse_train)

    # Calculate and define testing metrics
    y_pred_test = weekly_forecast['yhat'][split_point:]
    mae_test = mean_absolute_error(test['y'], y_pred_test)
    mse_test = mean_squared_error(test['y'], y_pred_test)
    rmse_test = np.sqrt(mse_test)

    # Training metrics
    training_metrics = [
        ["MAE (Training)", mae_train],
        ["MSE (Training)", mse_train],
        ["RMSE (Training)", rmse_train]
    ]

    # Testing metrics
    testing_metrics = [
        ["MAE (Testing)", mae_test],
        ["MSE (Testing)", mse_test],
        ["RMSE (Testing)", rmse_test]
    ]

  

    # Display the forecasted data
    st.subheader("Forecasted Demand Data:")
    st.write(weekly_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Plot the forecast
    weekly_fig1 = weekly_model.plot(weekly_forecast)
    plt.title('Weekly Orders Forecast')
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')

    st.subheader("Weekly Orders Forecast Plot:")
    st.pyplot(weekly_fig1)

    weekly_fig2 = weekly_model.plot_components(weekly_forecast)

    st.subheader("Weekly Orders Forecast Components:")
    st.pyplot(weekly_fig2)

    st.subheader("Actual vs Predicted Weekly Orders")
    plt.figure(figsize=(15, 6))
    plt.plot(train['ds'], train['y'], label='Training Data', color='blue')
    plt.plot(test['ds'], test['y'], label='Actual Test Data', color='orange')
    plt.plot(train['ds'], y_pred_train, label='Predicted Training Data', color='red', linestyle='--')
    plt.plot(test['ds'], y_pred_test, label='Predicted Test Data', color='green', linestyle='--')
    plt.title('Actual vs Predicted Weekly Orders')
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')
    plt.legend()
    plt.grid(True)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("Actual vs Predicted Weekly Orders Plot:")
    st.pyplot(plt.gcf()) 

    weekly_forecast = weekly_forecast.merge(train[['ds', 'y']], on='ds', how='left')
    weekly_forecast = weekly_forecast.merge(test[['ds', 'y']], on='ds', how='left', suffixes=('', '_test'))
    weekly_forecast['y'].fillna(weekly_forecast['y_test'], inplace=True)
    weekly_forecast.drop(columns='y_test', inplace=True)

    # Calculate weekly standard deviation of actual demand
    weekly_forecast['std_dev'] = weekly_forecast['y'].rolling(window=7).std()

    # Calculate weekly safety stock
    weekly_forecast['safety_stock'] = 1.65 * weekly_forecast['std_dev'] * np.sqrt(1)

    # Calculate weekly average demand from actual data
    weekly_forecast['avg_weekly_demand'] = weekly_forecast['y'].rolling(window=7).mean()

    # Calculate weekly reorder point
    weekly_forecast['reorder_point'] = (weekly_forecast['avg_weekly_demand'] * 1) + weekly_forecast['safety_stock']

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(weekly_forecast['ds'], weekly_forecast['y'], label='Actual Demand', color='blue')
    plt.plot(weekly_forecast['ds'], weekly_forecast['yhat'], label='Predicted Demand', color='green')
    plt.plot(weekly_forecast['ds'], weekly_forecast['safety_stock'], label='Safety Stock', color='red', linestyle='--')
    plt.plot(weekly_forecast['ds'], weekly_forecast['reorder_point'], label='Reorder Point', color='orange', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Demand with Safety Stock and Reorder Point')
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')

    st.subheader("Actual vs Predicted Demand with Safety Stock and Reorder Point")
    st.pyplot()

    # Calculate weekly standard deviation of forecasted demand
    weekly_forecast['forecasted_std_dev'] = weekly_forecast['yhat'].rolling(window=7).std()

    # Calculate weekly safety stock for forecasted demand
    weekly_forecast['forecasted_safety_stock'] = 1.65 * weekly_forecast['forecasted_std_dev'] * np.sqrt(1)

    # Calculate weekly average demand from forecasted data
    weekly_forecast['forecasted_avg_weekly_demand'] = weekly_forecast['yhat'].rolling(window=7).mean()

    # Calculate weekly reorder point for forecasted demand
    weekly_forecast['forecasted_reorder_point'] = (weekly_forecast['forecasted_avg_weekly_demand'] * 1) + weekly_forecast['forecasted_safety_stock']

    # Create a new Matplotlib figure for the plot
    plt.figure(figsize=(20, 12))
    plt.plot(weekly_forecast['ds'], weekly_forecast['y'], label='Actual Demand', color='blue')
    plt.plot(weekly_forecast['ds'], weekly_forecast['yhat'], label='Predicted Demand', color='green')
    plt.plot(weekly_forecast['ds'], weekly_forecast['safety_stock'], label='Safety Stock (Actual)', color='red', linestyle='--')
    plt.plot(weekly_forecast['ds'], weekly_forecast['forecasted_safety_stock'], label='Safety Stock (Forecasted)', color='purple', linestyle='--')
    plt.plot(weekly_forecast['ds'], weekly_forecast['reorder_point'], label='Reorder Point (Actual)', color='orange', linestyle='--')
    plt.plot(weekly_forecast['ds'], weekly_forecast['forecasted_reorder_point'], label='Reorder Point (Forecasted)', color='pink', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Demand with Safety Stock and Reorder Point')
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')

    st.subheader("Actual vs Predicted Demand with Safety Stock and Reorder Point")
    st.pyplot()

    st.write(weekly_forecast[['ds', 'y', 'yhat', 'safety_stock', 'forecasted_safety_stock', 'reorder_point', 'forecasted_reorder_point']])

