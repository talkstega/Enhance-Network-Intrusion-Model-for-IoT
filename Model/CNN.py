import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, Dropout, Activation, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler

filepath = "iot23_combined.csv"
dframe = pd.read_csv(filepath)
print(dframe)

del dframe['Unnamed: 0']
print(dframe['label'].value_counts())

X = dframe[['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp', 'proto_tcp', 'proto_udp', 'conn_state_OTH', 'conn_state_REJ', 'conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR', 'conn_state_RSTRH', 'conn_state_S0', 'conn_state_S1', 'conn_state_S2', 'conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR']].values
print(X.shape)

Y = pd.get_dummies(df['label']).values
print(Y.shape)
print(X)

print(dframe)

scaler = MinMaxScaler()
scaler.fit(X)
normalized_x = scaler.transform(X)
normalized_x
print(normalized_x.shape)

scaler.fit(Y)
normalized_y = scaler.transform(Y)
normalized_y

X_train, X_test, Y_train, Y_test = train_test_split(normalized_x, normalized_y, random_state=10, test_size=0.2)
X_train.shape

model = Sequential()

#set activation=relu rectified linear unit f(x)=max(0,x).
model.add(Dense(2000, activation='relu',input_dim=24))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

start = time.time()
print('program start...')
print()

#verbose refers to a particular setting used when training and validating models. 
#When verbose is turned on, the algorithm will provide more detailed information 
#about its progress as your model iterates through the training process.
history = model.fit(X_train, Y_train, epochs = 10, batch_size=256, validation_data=(X_test,Y_test),verbose=1)

print()
end = time.time()
print('program end...')
print()
print('time cost: ')
print(end - start, 'seconds')
