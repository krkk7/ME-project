#======================= IMPORT PACKAGES =============================
"Import Libaries "
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.layers import Conv1D, MaxPool1D, Flatten, Input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

#===================== DATA SELECTION ==============================

print("==================================================")
print(" Attention Network for In-Vehicle Intrusion Detection Attack ")
print("==================================================")
print("1.Data Selection ")
dataframe=pd.read_csv("dataset.csv")
print(dataframe.shape)
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()

#===================== DATA PREPROCESSING ==============================


#checking  missing values 
print("2.Data Pre processing  ")
print("==================================================")
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
#----------------------------------------------------------------------

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print() 
print(dataframe_2['flag'].unique().shape)
encoder = LabelEncoder()
scaler = StandardScaler()
onehot = OneHotEncoder()

cols_label_encoder = ['service', 'flag', 'class']
cols_scaler = []
cols_onehot = ['protocol_type']

for col in cols_label_encoder:
    dataframe_2[col] = encoder.fit_transform(dataframe_2[col])

dataframe_2 = pd.get_dummies(dataframe_2, columns=cols_onehot)

sns.countplot(x=dataframe_2['class'], data=dataframe_2)
plt.show()

#===================== DROPPING ON LABEL DATA  ==============================


X = dataframe_2.drop(labels='class', axis=1)
y = dataframe_2['class']

#===================== DATA SPLITTING  ==============================

print("4.Data Splitting  ")
print("==================================================")
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 4)
print("X_train Shapes ",x_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",x_test.shape)
print("y_test Shapes ",y_test.shape)

#===================== DATA CLASSIFICATION   ==============================

print("5.Data Classification ")
print("==================================================")
print("CNN ALGORITHM  -- PROPOSED-METHOD ")

"CNN Algorithm "
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import Sequential
input_shape = (43,)

model = Sequential([
    Dense(128, input_shape=input_shape),
    Activation('relu'),
    Dense(64, kernel_regularizer=l2(0.01)),
    Activation('relu'),
    Dense(64, kernel_regularizer=l2(0.01)),
    Activation('relu'),
    Dense(64, kernel_regularizer=l2(0.01)),
    Activation('relu'),
    Dropout(0.25),
    Dense(1),
    Activation('sigmoid')
])

model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=32, epochs=25, validation_data=(x_test, y_test), callbacks=[EarlyStopping(patience=3)])

# Get the training and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the training and validation accuracy from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# Plot the training and validation accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()
CNN_ACC=model.evaluate(x_train,y_train,verbose=1)[1]*100

CNN_prediction = model.predict(x_test)
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print(" CNN ")
print()
print(metrics.classification_report(y_test,CNN_prediction.round()))
print()
print("CNN   Accuracy is:",CNN_ACC,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, CNN_prediction.round())
print(cm2)
print("-------------------------------------------------------")
print()


#------------------------------------------------------------------------------
print("==================================================")
print("LSTM  ALGORITHM  -- EXISTING-METHOD ")
# x_train=np.expand_dims(x_train, axis=2)
# x_test=np.expand_dims(x_test, axis=2)
# y_train=np.expand_dims(y_train,axis=1)
# y_test=np.expand_dims(y_test,axis=1)


"LSTM Algorithm "

nb_out = 1
model = Sequential()
model.add(LSTM(input_shape=(43, 1), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
history=model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

print(model.summary())
# fit the model

model.fit(x_train, y_train, epochs=5, batch_size=20, verbose=1)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), callbacks=[EarlyStopping(patience=3)])

Result_3=model.evaluate(x_train,y_train,verbose=1)[1]*100
#from sklearn.metrics import accuracy_score
from sklearn import metrics

LSTM_prediction = model.predict(x_test)
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print(" LSTM")
print()
print(metrics.classification_report(y_test,LSTM_prediction.round()))
print()
print("LSTM  Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, LSTM_prediction.round())
print(cm2)
print("-------------------------------------------------------")
print()


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#====================== COMPARISION ===============================
if(CNN_ACC>Result_3):
    print("==========================")
    print()
    print("Convolutional Neural Network  is efficient")
    print()
    print("=========================")
else:
    print("==========================")
    print()
    print("LSTM  is efficient")
    print()
    print("=========================")

#----------------------------------------------------------------
print("5.Data Prediction")
print("==================================================")
CNN_prediction=CNN_prediction>0.5
from easygui import *
Key = "Enter the In-Vehicle Intrusion Detection "
  
# window title
title = "In-Vehicle Intrusion Detection system  Id "
# creating a integer box
str_to_search1 = enterbox(Key, title)
input = int(str_to_search1)

import tkinter as tk
if (CNN_prediction[input] ==0 ):
    print("NON ANOMALY")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Non ANOMALY ")
    tk.mainloop()
elif (CNN_prediction[input] ==1 ):
    print("ANOMALY ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "ANOMALY ")
    tk.mainloop()
    
    