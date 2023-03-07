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
#RnnWorkDataOH_NTB, RnnWeekDataOH_NTB = rnnMatLoaderNTB(PersonId = '10.131.2.1') # NTB means No Time Base
# model parameters :
#print('RnnWorkDataOH_NTB.shape =>',RnnWorkDataOH_NTB.shape)                     # (m,tx,n_values)
#print('RnnWeekDataOH_NTB.shape =>',RnnWeekDataOH_NTB.shape)                     # (m,tx,8)
# ------------------------------------------------------------------------------------------------------
#reconstracted data :
Tx = 161
n_values = 243
array = np.arange(Tx )
array = np.stack((array,array))
RnnWorkDataOH_NTB = tf.one_hot(array,n_values, on_value = 1.0, off_value = 0.0, axis =-1)
print('RnnWorkDataOH_NTB.shape =>',RnnWorkDataOH_NTB.shape)                     # (m,tx,n_values)
# ------------------------------------------------------------------------------------------------------
X, Y = xy(RnnWorkDataOH_NTB)
#X = X[0:2, :, :]  # shape => (23, 162, 243) to (2, 162, 243)
#Y = Y[:, 0:2, :]  # shape => (162, 23, 243) to (162, 2, 243)
n_values = X.shape[-1]  # number of work values =>
Tx = X.shape[1]         # Sequence length  =>
m = X.shape[0]          # number of training examples  =>
n_a = 32  # number of dimensions for the hidden state of each LSTM cell. hyper parameters
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
'''
prediction =>
[[  0   0]
 [  1   1]
 [  2   2]
 [  3   3]
 [  4   4]
 [  5   5]
 [  6   6]
 [  7   7]
 [  8   8]
 [  9   9]
 [ 10  10]
 [ 11  11]
 [ 12  12]
 [ 13  13]
 [ 14  14]
 [ 15  15]
 [ 16  16]
 [ 17  17]
 [ 18  18]
 [ 19  19]
 [ 20  20]
 [ 21  21]
 [ 22  22]
 [ 23  23]
 [ 24  24]
 [ 25  25]
 [ 26  26]
 [ 27  27]
 [ 28  28]
 [ 29  29]
 [ 30  30]
 [ 31  31]
 [ 32  32]
 [ 33  33]
 [ 34  34]
 [ 35  35]
 [ 36  36]
 [ 37  37]
 [ 38  38]
 [ 39  39]
 [ 40  40]
 [ 41  41]
 [ 42  42]
 [ 43  43]
 [ 44  44]
 [ 45  45]
 [ 46  46]
 [ 47  47]
 [ 48  48]
 [ 49  49]
 [ 50  50]
 [ 51  51]
 [ 52  52]
 [ 53  53]
 [ 54  54]
 [ 55  55]
 [ 56  56]
 [ 57  57]
 [ 58  58]
 [ 59  59]
 [ 60  60]
 [ 61  61]
 [ 62  62]
 [ 63  63]
 [ 64  64]
 [ 65  65]
 [ 66  66]
 [ 67  67]
 [ 68  68]
 [ 69  69]
 [ 70  70]
 [ 71  71]
 [ 72  72]
 [ 73  73]
 [ 74  74]
 [ 75  75]
 [ 76  76]
 [ 77  77]
 [ 78  78]
 [ 79  79]
 [ 80  80]
 [ 81  81]
 [ 82  82]
 [ 83  83]
 [ 84  84]
 [ 85  85]
 [ 86  86]
 [ 87  87]
 [ 88  88]
 [ 89  89]
 [ 90  90]
 [ 91  91]
 [ 92  92]
 [ 93  93]
 [ 94  94]
 [ 95  95]
 [ 96  96]
 [ 97  97]
 [ 98  98]
 [ 99  99]
 [100 100]
 [101 101]
 [102 102]
 [103 103]
 [104 104]
 [105 105]
 [106 106]
 [107 107]
 [108 108]
 [109 109]
 [110 110]
 [111 111]
 [112 112]
 [113 113]
 [114 114]
 [115 115]
 [116 116]
 [117 117]
 [118 118]
 [119 119]
 [120 120]
 [121 121]
 [122 122]
 [123 123]
 [124 124]
 [125 125]
 [126 126]
 [127 127]
 [128 128]
 [129 129]
 [130 130]
 [131 131]
 [132 132]
 [133 133]
 [134 134]
 [135 135]
 [136 136]
 [137 137]
 [138 138]
 [139 139]
 [140 140]
 [141 141]
 [142 142]
 [143 143]
 [144 144]
 [145 145]
 [146 146]
 [147 147]
 [148 148]
 [149 149]
 [150 150]
 [151 151]
 [152 152]
 [153 153]
 [154 154]
 [155 155]
 [156 156]
 [157 157]
 [158 158]
 [159 159]
 [160 160]
 [  0   0]]'''
# nice ;)


