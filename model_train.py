# Finally Keras Model !!
# imports as always
import time
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
from data_analyst import *
from models import *

start = time.time()

# parameters
intuition = False
write = False
testPhase = False
showSummary = False
figShow = True
verbose = 1
n_a = 64                # number of dimensions for the hidden state of each LSTM cell. hyper parameters
inputs = 12             # 12 models with 1 to 12 inputs ...
outputs = 12
day = 67
learning_rate = 0.01
epochs = 13
if testPhase:           # epochs in test phase
    epochs = 1
else:
    epochs = epochs     # epochs in normal training phase
if write:
    small_writer(tx=30)


X_work, Y_work, X_week, Y_week = reader('10.130.2.1')
X, Y = X_work, Y_work


if figShow:
    X_plotter_2D(day, X_work)
    Y_plotter_2D(day, Y_work)


# extracting parameters from data
n_values = X.shape[-1]                  # number of work values => ...
Tx = X.shape[1]                         # Sequence length  => ...
m = X.shape[0]                          # number of training examples  => ...
print('model parameters :')
print('n_values =>', n_values)
print('tx =>', Tx)
print('m =>', m)
print('n_a =>', n_a)


# this part made for sequence generation uses a for-loop
# Please read => readMe/Sequence_generation.md
reshaper = Reshape((1, n_values), name='myReshaper')
LSTM_cell = LSTM(n_a, return_state=True, name='myLSTM_cell')
densor = Dense(n_values, activation='softmax', name='myDensor')


if intuition:
    keras_instructor()  # to get more intuition


# Create the model object
# Run the following cell to define your model
# We will use tx = 30 .
# This cell may take a few seconds to run
model = trainable_model(Tx=Tx, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)


if showSummary:
    model.summary()


print('model.input_shape => \n', model.input_shape)
print('model.output_shape => \n', model.output_shape)


# optimizer ...
opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)


# compile ...
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# hidden cell initialization
n_a = LSTM_cell.units    # number of hidden cell
a0 = np.zeros((m, n_a))  # ...
c0 = np.zeros((m, n_a))  # ...


# and fit ...
print('learning process ...')
print('X.shape =>', X.shape)
print('Y.shape =>', Y.shape)
history = model.fit([np.array(X), a0, c0], list(Y), epochs=epochs, verbose=verbose)


# plotting learning curves ...
print(f"loss at epoch 1: {history.history['loss'][0]}")
print(f"loss at epoch " + str(len(history.epoch))+f" : {history.history['loss'][len(history.epoch) - 1]}")
if figShow:
    plt.plot(history.history['loss'])
    plt.show()

# nice

# Generating Sequence(single input without beam search)
# Please read => readMe/Generating_Sequence(single_input_without_beam_search)
# Why should we use these?
# - tf.math.argmax()
# - tf.one_hot()
# - RepeatVector()
# Pleas read => readMe/ tf_methods.md


uni_input_inference_model_test = uni_input_inference_model(LSTM_cell, densor, Ty=inputs)
print('uni_input_inference_model_test.input_shape => \n', uni_input_inference_model_test.input_shape)
print('uni_input_inference_model_test.output_shape => \n', uni_input_inference_model_test.output_shape)
uni_input_inference_model_test.save("saved_models/test_models/uni_input_inference_model_test.h5")
print("written to saved_models/test_models/uni_input_inference_model_test.h5")


if showSummary:
    uni_input_inference_model_test.summary()


multi_input_sequence_inference_model_test = multi_input_sequence_inference_model(
    LSTM_cell, densor, reshaper, Ty=outputs, Tx=inputs)
testX = np.expand_dims(X[1, 0:outputs, :], axis=0)                # ...
testX2 = np.stack((np.squeeze(testX), np.squeeze(testX)))  # (m,tx,n_values) (2, 10, 293)
testA2 = np.zeros((2, n_a))                                # (m,n_values) (2, 32)
testC2 = np.zeros((2, n_a))                                # (m,n_values) (2, 32)
prediction = multi_input_sequence_inference_model_test.predict([testX2, testA2, testC2])
print('prediction.shape => ', np.array(prediction).shape)  # (Ty, m, n_values) (50, 2, 293)


multi_input_sequence_inference_model_test.save(f"saved_models/test_models/model{outputs}Inputs.h5")
print(f"written to saved_models/test_models/model{outputs}Inputs.h5")


# saving models to h5
maxInput = inputs
for number_of_inputs in range(1, maxInput + 1):
    model = multi_input_sequence_inference_model(LSTM_cell, densor, reshaper, Ty=1, Tx=number_of_inputs)
    model.save(f"saved_models/models/model{number_of_inputs}Inputs.h5")
    print(f"written to saved_models/models/model{number_of_inputs}Inputs.h5")


# saving models to h5
maxInput = inputs
for number_of_inputs in range(1, maxInput + 1):
    model = multi_input_HiddenCell_sequence_inference_model(LSTM_cell, densor, reshaper, Ty=1, Tx=number_of_inputs)
    model.save(f"saved_models/hc_models/model{number_of_inputs}Inputs_HC.h5")  # HC means HiddenCell
    print(f" written to saved_models/hc_models/model{number_of_inputs}Inputs_HC.h5")


end = time.time()
print("The time of execution program =>", (end-start) * 10**3, "ms")
print("The time of execution program =>", (end-start) / 60, "minutes")


# Shahbaz, Mohammad Javad and Kasra at Chargoon
