9

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from src.utils import create_test, make_forecasts, inverse_transform
import pandas as pd

#process data

vietnam_file_path = r"data/influA_vietnam_temp_month.csv"
japan_file_path = r"data/influA_japan_temp_month.csv"
vietnam_weight_path = r"weights/vietnam/best_month_model.weights.h5"
japan_weight_path = r"weights/japan/best_month_model.weights.h5"


with open( './weights/vietnam/config.json') as f:
    config_vietnam = json.load(f)

with open( './weights/japan/config.json') as f:
    config_japan = json.load(f)
model_vietnam, testX_vietnam, scaler_vietnam, test_month_vietnam = create_test(config_vietnam, vietnam_file_path, vietnam_weight_path)
model_japan, testX_japan, scaler_japan, test_month_japan = create_test(config_japan, japan_file_path, japan_weight_path)

# Title of the app
st.title("Bộ dữ liệu bạn muốn chọn")

# Select box input
options = ["Việt Nam", "Nhật Bản"]
selected_option = st.selectbox("Select an option:", options)

# Button to trigger an action
if st.button("Forecast"):
    option_index = options.index(selected_option)
    #Vietnam
    if option_index == 0:
        model = model_vietnam
        testX= testX_vietnam
        test_month = test_month_vietnam
        scaler = scaler_vietnam
    else:
        model = model_japan
        testX = testX_japan
        test_month = test_month_japan
        scaler = scaler_japan
    
    forecasts = make_forecasts(testX, model)
    # invert predictions
    forecasts = inverse_transform(forecasts,scaler)
    # evaluate_forecasts(testY_[:,0].reshape(-1), forecasts[:, 0].reshape(-1))

    #Doan nay dang khac voi ben kia
    new_shape = (forecasts.shape[0] * forecasts.shape[1], forecasts.shape[2])
    result = forecasts.reshape(new_shape)[:,0].flatten()
    result = [round(a) for a in result]


    # st.write("Result: ", forecasts.reshape(new_shape)[:,0].flatten())
    # st.map(forecasts.reshape(new_shape)[:,0].flatten())
    chart_data = pd.DataFrame(
     {"case": result, 'Month': test_month}
     )

    st.line_chart(chart_data, x_label='Month', y_label='case', x='Month', y='case')
