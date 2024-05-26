from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import cm

# import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import QuantileTransformer , PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from keras import optimizers
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten
from keras.layers import Dense, Dropout, Activation, LSTM
# from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from keras.models import load_model
import pickle
import random

# from flask_mysqldb import MySQL
from flask import request, jsonify
from flask import Response, json, request, jsonify, Flask

app = Flask(__name__)

def create_dataset(X, look_back=1):
    data = []
    for i in range(len(X)-look_back-1):
        data.append(X[i:(i+look_back)])
    return np.array(data)


model = load_model('LSTM_with_lookback_1.h5')
test_file = "test_FD001.txt"
columns = ["Section-{}".format(i)  for i in range(26)]
df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns

RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])
# max_cycles[0] = max_cycles[0] - 21


df_test.drop(df_test[["Section-0",
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"]], axis=1 , inplace=True)


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
pt = PowerTransformer()
df_test = pt.fit_transform(df_test)
df_test=np.nan_to_num(df_test)

test_data = []
i = 0
count = 0
while i < len(df_test):
    temp = []
    j = int(max_cycles[count])
    count = count+1
    if j == 0:
        break
    while j!=0:
        temp.append(df_test[i])
        i=i+1
        j=j-1
    test_data.append(temp)

y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))
future_cycles = 20

def get_scores():
    predictions = []
    for i in range(len(y_new)):
        test_model = VAR(y_new[i])
        test_model_fit = test_model.fit()
        test_pred = test_model_fit.forecast(test_model_fit.endog, steps=future_cycles)
        predictions.append(test_pred)


    lstm_pred = []
    for i in range(len(predictions)):
        predictions[i] = np.reshape(predictions[i],(predictions[i].shape[0],1,predictions[i].shape[1]))
        lstm_pred.append(model.predict(predictions[i]))

    lstm_pred = list(lstm_pred)
    for i in range(len(lstm_pred)):
        lstm_pred[i] = list(lstm_pred[i])
    for i in range(len(lstm_pred)):
        for j in range(len(lstm_pred[i])):
            lstm_pred[i][j] = int(lstm_pred[i][j])

    health_scores = []
    for i in range(len(lstm_pred)):
        temp_health_scores = []
        for j in range(len(lstm_pred[i])):
            if lstm_pred[i][j] >= 100:
                temp_health_scores.append(1)
            else:
                temp_health_scores.append(float(lstm_pred[i][j])/100)
        health_scores.append(temp_health_scores)

    random_index_array = random.sample(range(0, len(test_data)), len(test_data))
    selected_health_scores = []
    # counter = 0 
    i = 0
    while i< len(random_index_array):
        constant_list = True
        for j in range(1,len(health_scores[random_index_array[i]])):
            if health_scores[random_index_array[i]][j] != health_scores[random_index_array[i]][0]:
                constant_list = False
                # counter = counter + 1
                break
        if (sorted(health_scores[random_index_array[i]],reverse=True) == health_scores[random_index_array[i]]) and constant_list==False:
            selected_health_scores.append(health_scores[random_index_array[i]])
        i = i + 1
    
    i=0
    while len(selected_health_scores) != 20:
        constant_list = True
        for j in range(1,len(health_scores[random_index_array[i]])):
            if health_scores[random_index_array[i]][j] != health_scores[random_index_array[i]][0]:
                constant_list = False
                # counter = counter + 1
                break
        if constant_list == True:
            selected_health_scores.append(health_scores[random_index_array[i]])
        i = i + 1
    to_be_inserted = []
    for i in range(len(selected_health_scores[0])):
        temp=[]
        for j in range(len(selected_health_scores)):
            temp.append(selected_health_scores[j][i])
        to_be_inserted.append(temp)
    return to_be_inserted

class current:
    def __init__(self):
        self.curr=0
    def incr_curr(self):
        self.curr=self.curr+1
    def get_curr(self):
        return self.curr
    def restart_curr(self):
        self.curr = 0
class_curr = current()
to_be_inserted = get_scores()

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/restart',methods=['GET', 'POST'])
def restart():
    class_curr.restart_curr()
    return render_template("home.html")

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    id = class_curr.get_curr()
    class_curr.incr_curr()
    if id < future_cycles:
        return render_template('temporary.html',pred=to_be_inserted[id])
    else :
        return render_template('finish.html',pred=to_be_inserted[19])

if __name__ == '__main__':
    app.run(debug=True)
