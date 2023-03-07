import numpy as np
from numpy import log
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow import random
from instructor import *
from data_utils import *
from graphic_utils import *
# ------------------------------------------------------------------------------------------------------
RnnWorkDataOH_NTB, RnnWeekDataOH_NTB = rnnMatLoaderNTB(PersonId = '10.131.2.1') # NTB means No Time Base
# model parameters :
print('RnnWorkDataOH_NTB.shape =>',RnnWorkDataOH_NTB.shape)                     # (m,tx,n_values)
print('RnnWeekDataOH_NTB.shape =>',RnnWeekDataOH_NTB.shape)                     # (m,tx,8)
# ------------------------------------------------------------------------------------------------------
#reconstracted data :
'''tx = 161
n_values = 243
array = np.arange(tx )
array = np.stack((array,array))
RnnWorkDataOH_NTB = tf.one_hot(array,n_values, on_value = 1.0, off_value = 0.0, axis =-1)
print('RnnWorkDataOH_NTB.shape =>',RnnWorkDataOH_NTB.shape)                     # (m,tx,n_values)'''
# ------------------------------------------------------------------------------------------------------
X, Y = xy(RnnWorkDataOH_NTB)
#X = X[0:2, :, :]       # shape => (23, 162, 243) to (2, 162, 243)
#Y = Y[:, 0:2, :]       # shape => (162, 23, 243) to (162, 2, 243)
n_values = X.shape[-1]  # number of work values =>
Tx = X.shape[1]         # Sequence length  =>
m = X.shape[0]          # number of training examples  =>
n_a = 128               # number of dimensions for the hidden state of each LSTM cell. hyper parameters
print('X.shape =>', X.shape)
print('Y.shape =>', Y.shape)
print('model parameters :')
print('n_values =>', n_values)
print('tx =>', Tx)
print('m =>', m)
print('n_a =>', n_a)
print(line)
# ------------------------------------------------------------------------------------------------------
# this part made for sequence generation uses a for-loop
# Please read => readMe/Sequence_generation.md
reshaper = Reshape((1, n_values), name='myReshaper')
LSTM_cell = LSTM(n_a, return_state = True, name='myLSTM_cell')
densor = Dense(n_values, activation='softmax',name='myDensor')
# ---------------------------------------------------------------------
def trmModel(Tx, LSTM_cell, densor, reshaper):
    n_values = densor.units
    n_a = LSTM_cell.units
    X = Input(shape=(Tx, n_values), name='X')  # (None, tx, n_values)
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []
    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)  # (None, 1, n_values)
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model
# ---------------------------------------------------------------------
print('creating model ...')
model = trmModel(Tx = Tx , LSTM_cell = LSTM_cell, densor = densor, reshaper = reshaper)
print('model.output_shape => \n',model.output_shape)
# ---------------------------------------------------------------------
# hidden cell initialization
n_a = LSTM_cell.units   # number of hidden cell
a0 = np.zeros((m, n_a)) # ...
c0 = np.zeros((m, n_a)) # ...
# ---------------------------------------------------------------------
# optimizer ...
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, decay=0.01)
# ---------------------------------------------------------------------
# compile ...
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# ---------------------------------------------------------------------
# and fit ...
print('learning process ...')
history = model.fit([np.array(X), a0, c0], list(Y), epochs = 30, verbose = 1)
# ---------------------------------------------------------------------
print('plotting loss ...')
print(f"loss at epoch 1: {history.history['loss'][0]}")
print(f"loss at epoch " + str(len(history.epoch))+f" : {history.history['loss'][len(history.epoch) - 1]}")
plt.plot(history.history['loss'])
plt.show()
# ---------------------------------------------------------------------
m = 2
testX2 = X[0:m , :, :]                                    # (m,tx,n_values) (2, 5, 243)
testA2 = np.zeros((m, n_a))                               # ...
testC2 = np.zeros((m, n_a))                               # ...
# ---------------------------------------------------------------------
print('prediction ...')
prediction = np.array(model.predict([testX2, testA2, testC2]))
# ---------------------------------------------------------------------
def ohToPred(array):
    return(np.argmax(array, axis=-1))
print('prediction.shape =>',prediction.shape)
print('prediction =>\n',ohToPred(prediction))