# Import all the required packages
from flask import Flask, jsonify, request, make_response, Response
import os
from dateutil import *
import dateutil.relativedelta
from datetime import date
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS
import requests

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import json

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    # Creating the image path for model loss, LSTM generated image and all issues data image
    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type + "_" + repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type + "_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_" + repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    



    created_at = data_frame['created_at']
    shah = pd.to_datetime(
        pd.Series(created_at), format='%Y/%m/%d')
    shah.index = shah.dt.to_period('m')
    shah_create = []
    shah = shah.groupby(level=0).size()
    shah = shah.reindex(pd.period_range(
        shah.index.min(), shah.index.max(), freq='m'), fill_value=0)
    shah_dict = shah.to_dict()
    for key in shah_dict.keys():
        array = [str(key), shah_dict[key]]
        shah_create.append(array)


    closed_at = data_frame['closed_at'].sort_values(ascending=True)
    month_issue_closed = pd.to_datetime(
        pd.Series(closed_at), format='%Y/%m/%d')
    month_issue_closed.index = month_issue_closed.dt.to_period('m')
    shah_close = []
    month_issue_closed = month_issue_closed.groupby(level=0).size()
    month_issue_closed = month_issue_closed.reindex(pd.period_range(
        month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'), fill_value=0)
    month_issue_closed_dict = month_issue_closed.to_dict()
    for key in month_issue_closed_dict.keys():
        array = [str(key), month_issue_closed_dict[key]]
        shah_close.append(array)


    plt.figure(figsize=(10, 5))
    k = []
    bottom_values = []
    for i in range(len(shah_create)):
        k.append(shah_create[i][0])
        bottom_values.append(shah_create[i][1])
    top_values = []
    for i in range(len(shah_close)):
        top_values.append(shah_close[i][1])
    plt.bar(k, bottom_values, color = 'black')
    plt.bar(k, top_values, bottom = bottom_values, color='gold')
    plt.legend(["created issues monthly", "closed issues monthly"])
    plt.xticks(rotation=70)
    plt.title('stack bar (column) chart.')
    STACK = "stack_bar" + "_" + repo_name + ".png"
    STACK_URL = BASE_IMAGE_PATH + STACK
    plt.savefig(LOCAL_IMAGE_PATH + STACK)

    
    data_frame['created_at'] = pd.to_datetime(data_frame['created_at'], errors='coerce')
    shah_df_week = data_frame.groupby(data_frame['created_at'].dt.day_name()).size()
    shah_df_week = pd.DataFrame({'on':shah_df_week.index, 'Count':shah_df_week.values})
    x = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday' ]
    shah_df_week = shah_df_week.groupby(['on']).sum().reindex(x)
    plt.figure(figsize=(10, 5))
    plt.plot(shah_df_week['Count'])
    plt.title('Created for week days.')
    plt.ylabel('Issues No.')
    plt.xlabel('time')

    CREATE_WEEK = "create_week" + "_" + repo_name + ".png"
    CREATE_WEEK_URL = BASE_IMAGE_PATH + CREATE_WEEK
    plt.savefig(LOCAL_IMAGE_PATH + CREATE_WEEK)
    
    data_frame['closed_at'] = pd.to_datetime(data_frame['closed_at'], errors='coerce')
    shah_df_week = data_frame.groupby(data_frame['closed_at'].dt.day_name()).size()
    shah_df_week = pd.DataFrame({'cl_on':shah_df_week.index, 'Count':shah_df_week.values})
    shah_df_week = shah_df_week.groupby(['cl_on']).sum().reindex(x)
    plt.figure(figsize=(10, 5))
    plt.plot(shah_df_week['Count'])
    plt.title('Closed for week days.')
    plt.ylabel('Issues No.')
    plt.xlabel('time')
    
    CLOSE_WEEK = "close_week" + "_" + repo_name + ".png"
    CLOSE_WEEK_URL = BASE_IMAGE_PATH + CLOSE_WEEK
    plt.savefig(LOCAL_IMAGE_PATH + CLOSE_WEEK)
    

    data_frame['closed_at'] = pd.to_datetime(data_frame['closed_at'], errors='coerce')
    shah_df_month = data_frame.groupby(data_frame['closed_at'].dt.month_name()).size()
    shah_df_month = pd.DataFrame({'cl_on':shah_df_month.index, 'Count':shah_df_month.values})
    x = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    shah_df_month = shah_df_month.groupby(['cl_on']).sum().reindex(x)
    plt.figure(figsize=(10, 5))
    plt.plot(shah_df_month['Count'])
    plt.title('closed for months.')
    plt.xticks(rotation=70)
    plt.ylabel('Issues No.')
    plt.xlabel('time')

    CLOSE_MONTH = "close_month" + "_" + repo_name + ".png"
    CLOSE_MONTH_URL = BASE_IMAGE_PATH + CLOSE_MONTH
    plt.savefig(LOCAL_IMAGE_PATH + CLOSE_MONTH)

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    new_blob = bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)
    new_blob = bucket.blob(STACK)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + STACK)
    new_blob = bucket.blob(CREATE_WEEK)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CREATE_WEEK)
    new_blob = bucket.blob(CLOSE_WEEK)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CLOSE_WEEK)
    new_blob = bucket.blob(CLOSE_MONTH)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CLOSE_MONTH)
    bucket = client.get_bucket(BUCKET_NAME)

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,

        "lstm_generated_image_url": LSTM_GENERATED_URL,
        
        "all_issues_data_image": ALL_ISSUES_DATA_URL,
        
        "stackBarImageURL": STACK_URL,
        
        "weekCreatedImageURL": CREATE_WEEK_URL,
        
        "weekClosedImageURL": CLOSE_WEEK_URL,
        
        "monthClosedImageURL": CLOSE_MONTH_URL,
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)

# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
