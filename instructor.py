import numpy as np
from tensorflow import random
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector


def keras_instructor():
    # please run this Keras example to get more ituation
    inputs = random.normal([32, 10, 8])
    # lstm = LSTM(4)
    # ---------------------------------------------------------------------
    lstm = LSTM(4, return_sequences=True, return_state=True)
    whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
    print('whole_seq_output =>', whole_seq_output.shape)
    print('final_memory_state =>', final_memory_state.shape)
    print('final_carry_state =>', final_carry_state.shape)
    print('--------------------------------')
    # ---------------------------------------------------------------------
    lstm = LSTM(4, return_sequences=True, return_state=False)
    whole_seq_output = lstm(inputs)
    print('whole_seq_output =>', whole_seq_output.shape)
    print('--------------------------------')
    # ---------------------------------------------------------------------
    lstm = LSTM(4, return_sequences=False, return_state=True)
    lstm1, state_h, state_c, = lstm(inputs)
    print('lstm1 =>', lstm1.shape)
    print('state_h =>', state_h.shape)
    print('state_c =>', state_c.shape)
    print('--------------------------------')  # Note that lstm1 and state_h are same
    # ---------------------------------------------------------------------
    lstm = LSTM(4, return_sequences=False, return_state=False)
    output = lstm(inputs)
    print('output =>', output.shape)
    # ---------------------------------------------------------------------
    inputs = np.zeros(shape=(32, 10, 8))
    lstm = LSTM(4)
    output = lstm(inputs)
    print('output =>', output.shape)
