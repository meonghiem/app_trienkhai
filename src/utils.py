import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, TimeDistributed, Input, Dropout, GRU,concatenate, Bidirectional
from keras.saving import register_keras_serializable
from keras.optimizers import Adam
from keras.optimizers.schedules import CosineDecay

import tensorflow as tf


def read_data(file_path, num_features = 2, have_time = False):
    series_influ_A_df = read_csv(file_path, engine='python')
    series_influ_A_df = series_influ_A_df.rename(columns= {"Influenza A - All types of surveillance": "case"})

    # because since 2011-03-01 It was announced that the H1N12009 flu had been controlled and treated as regular seasonal flu and
    # since 2020-02-01, it's time for covid 
    
    series_influ_A_df = series_influ_A_df.loc [(series_influ_A_df['Month'] >='2011-04-01') & (series_influ_A_df['Month'] <='2020-02-01')]
    if not have_time:
        return series_influ_A_df.dropna()[["case", "temp", "dew", "tempmax", "humidity","tempmin","windspeed"][:num_features]]
    return series_influ_A_df.dropna()[["Month", "case", "temp", "dew", "tempmax", "humidity","tempmin","windspeed"][:num_features+1]]


def create_dataset(dataset, look_back=1, num_predict = 1):
    dataX, dataY = [], []
    for i in range(0, len(dataset)-look_back, num_predict):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back: i+ look_back +num_predict])
    return np.array(dataX), np.array(dataY)

def forecast(input, model):
    input = input.reshape(1, input.shape[0], input.shape[1])
    predicted = model.predict(input, verbose=0)
    return predicted[0]

def make_forecasts(test, model):
    forecasts = []
    inputStart = test[0]
    inputX = inputStart.reshape(1, inputStart.shape[0], inputStart.shape[1])
    for i in range(len(test)):
        predicted = forecast(inputX[i], model)
        forecasts.append(predicted)
        if i!= len(test)-1:
            inputXContinue = np.vstack((inputX[-1][predicted.shape[0]:], predicted))
            inputX = np.append(inputX, [inputXContinue], axis=0)
    return np.array(forecasts)

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def exponential_moving_average(data, span):
    return data.ewm(span=span, adjust=False).mean()

def prepare_data(df, is_smooth=[False, False], scaler =None):
    import pandas as pd
    df_ = pd.DataFrame()
    df_ = df.copy()
    is_ma, is_ema = is_smooth
    if is_ema:
        # Apply Moving Average Filter
        window_size = 2
        
        df_['case'] = exponential_moving_average(df_['case'], window_size)
        df_ = df_.dropna()
    if is_ma:
        # Apply Moving Average Filter
        window_size = 2
        
        df_['case'] = moving_average(df_['case'], window_size)
        df_ = df_.dropna()
    dataset = df_.values
    month_dataset = dataset[:,0]
    dataset = dataset[:,1:]
    if scaler:
        flattened_dataset = dataset.flatten()
        dataset_trans = scaler.fit_transform(flattened_dataset.reshape(-1,1))
        dataset = dataset_trans.reshape(dataset.shape)
    return  dataset, month_dataset
def inverse_transform(data, scaler):
    flattened_data = data.flatten()
    inverse_flattened_data = scaler.inverse_transform(flattened_data.reshape(-1,1))
    return inverse_flattened_data.reshape(data.shape)

def remove_outlier_IQR(df):

    q1=df.quantile(0.25)

    q3=df.quantile(0.75)

    IQR=q3-q1

    upper = df[~(df>(q3+1.5*IQR))].max()

    lower = df[~(df<(q1-1.5*IQR))].min()

    df = np.where(
        df > upper,
        np.nan,
        np.where(df < lower,np.nan,df)
        )
    return df

def remove_outlier_impute(df, impute_method= 'linear'):
    output_df = df.copy()
    if impute_method != "raw":
        output_df["case"] = remove_outlier_IQR(output_df['case'])
        match impute_method:
            case 'linear':
                output_df['case'] = output_df['case'].interpolate(method='linear')
            case 'spline':
                output_df['case'] = output_df['case'].interpolate(method='spline', order=3)
            case _:
                pass
    return output_df

learning_rate_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)

def build_model(input_shape, output_units, use_layer= "RNN", dropout=None, unit=64, optimizer='adam', load_weight=False):
    input = Input(shape=input_shape)
    num_predict = 8

    # GRU Model
    match use_layer:
        case "LSTM":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                LSTM(unit, activation='relu', return_sequences=True),
            ])
        case "GRU":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                GRU(unit, activation='relu', return_sequences=True),
            ])
        case "Bi-LSTM":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                Bidirectional(LSTM(unit, activation='relu', return_sequences=True)),
            ])
        case "Stacked-LSTM":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                LSTM(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                LSTM(unit, activation='relu', return_sequences=True),
            ])
        case "RNN-LSTM":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                SimpleRNN(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                LSTM(unit, activation='relu', return_sequences=True),
            ])
        case "LSTM-RNN":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                LSTM(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                SimpleRNN(unit, activation='relu', return_sequences=True),
            ])
        case "GRU-RNN":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                GRU(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                SimpleRNN(unit, activation='relu', return_sequences=True),
            ])
        case "RNN-GRU":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                SimpleRNN(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                GRU(unit, activation='relu', return_sequences=True),
            ])
        case "GRU-LSTM":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                GRU(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                LSTM(unit, activation='relu', return_sequences=True),
            ])
        case "LSTM-GRU":
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                LSTM(unit, activation='relu', return_sequences=True),
                Dropout(0.2),
                GRU(unit, activation='relu', return_sequences=True),
            ])
        # case "Seq2Seq":
        #     # nn_model = Sequential([
        #     #     # Input(input_shape=input_shape),
        #     #     LSTM(unit, activation='relu', return_sequences=True),
        #     #     Dropout(0.2),
        #     #     LSTM(unit, activation='relu', return_sequences=True),
        #     # ])
        #     encoder = SimpleRNN(unit,activation='relu', return_state=True)
        #     encoder_outputs, encoder_state = encoder(input)
        #     encoder_states = [encoder_state]
        #     #===============
        #     decoder_lstm = LSTM(unit, return_sequences=True, return_state=True)
        #     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        #     decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        #     decoder_outputs = decoder_dense(decoder_outputs)
        case _:
            nn_model = Sequential([
                # Input(input_shape=input_shape),
                SimpleRNN(unit, activation='relu', return_sequences=True),
            ])

    nn_output = nn_model(input)

    dense_layers = Sequential([
    TimeDistributed(Dense(units=output_units, activation='tanh' ))
    ])

    output = dense_layers(nn_output[:,-num_predict:])
    model = tf.keras.Model(inputs=input, outputs=output)
    if optimizer == 'adam':
        optimizer_func = Adam(learning_rate=learning_rate_schedule)
    if not load_weight:
        model.compile(optimizer=optimizer_func, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def load_model(trainX, trainY, config, weight_path):
    n_neuron = config["unit"]
    use_layer = config["use_layer"]
    model = build_model(
            input_shape=(trainX.shape[1], trainX.shape[2]),
            output_units=trainY.shape[2],
            use_layer=use_layer,
            unit=n_neuron,
            load_weight=True
            )
    model.load_weights(weight_path)
    return model

def create_test(config, file_path, weight_path):
    num_feature = config["num_feature"]
    impute_method = config["impute_method"]
    is_ma = config["is_ma"]
    look_back = config["look_back"]
    n_neuron = config["unit"]
    use_layer = config["use_layer"]

    #Load data set
    df = read_data(file_path, num_features=num_feature, have_time=True)
    output_df = remove_outlier_impute(df, impute_method)

    scaler = MinMaxScaler(feature_range=(-1,1))
    dataset, month = prepare_data(df = output_df, scaler=scaler, is_smooth=is_ma)

    num_predict = 8
    # look_back = 8
    test_size = num_predict
    length = len(dataset)
    dataset_length = length - (length -look_back) % num_predict
    rest = length - dataset_length
    dataset = dataset[rest:]
    train = dataset[: -test_size,:]
    test = dataset[-test_size - look_back:,:]

    month = month[rest:]
    test_month = month[-test_size:,]

    trainX, trainY = create_dataset(train, look_back, num_predict)
    testX, testY = create_dataset(test, look_back, num_predict)
    model = load_model(trainX, trainY, config, weight_path)
    return model, testX, scaler, test_month

