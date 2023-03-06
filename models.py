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


# Trainable Models ...
def trainable_model(Tx, LSTM_cell, densor, reshaper):
    """
    Implement the trmModel composed of tx LSTM cells where each cell is responsible
    for learning the following note based on the previous note and context.
    Each cell has the following schema:
            [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
    Arguments:
        tx -- length of the sequences in the corpus
        LSTM_cell -- LSTM layer instance
        densor -- Dense layer instance
        reshaper -- Reshape layer instance

    Returns:
        model -- a keras instance model with inputs [X, a0, c0]
    """
    # Get the shape of input values
    n_values = densor.units
    # ------------------------------------------------
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    # ------------------------------------------------
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values), name='X')  # (None, tx, n_values)
    # ------------------------------------------------
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    # -------------------------------------------------
    # Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    # -------------------------------------------------
    # Loop over tx
    for t in range(Tx):
        # ---------------------------------------------
        # Select the "t"th time step vector from X.
        x = X[:, t, :]  # (None, n_values)
        # Use reshaper to reshape x to be (1, n_values) (≈1 line)
        x = reshaper(x)  # (None, 1, n_values)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Add the output to "outputs"
        outputs.append(out)
    # -------------------------------------------------
    # Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


def trainable_small_model(Tx, LSTM_cell, densor, reshaper):
    """
    wht do i mean by small ? check the link below :
    https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
    It's just inspired by that but it's very different

    Implement the trmModel composed of tx LSTM cells where each cell is responsible
    for learning the following note based on the previous note and context.
    Each cell has the following schema:
            [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
    Arguments:
        tx -- length of the sequences in the corpus
        LSTM_cell -- LSTM layer instance
        densor -- Dense layer instance
        reshaper -- Reshape layer instance

    Returns:
        model -- a keras instance model with inputs [X, a0, c0]
    """
    # Get the shape of input values
    n_values = densor.units
    # ------------------------------------------------
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    # ------------------------------------------------
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values), name='X')  # (None, tx, n_values)
    # ------------------------------------------------
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    # -------------------------------------------------
    # Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    # -------------------------------------------------
    # Loop over tx
    for t in range(Tx):
        # ---------------------------------------------
        # Select the "t"th time step vector from X.
        x = X[:, t, :]  # (None, n_values)
        # Use reshaper to reshape x to be (1, n_values) (≈1 line)
        x = reshaper(x)  # (None, 1, n_values)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
    out = densor(a)
    # Add the output to "outputs"
    outputs.append(out)
    # -------------------------------------------------
    # Create model instance
    model = Model(inputs=[X, a0, c0], outputs=out)

    return model


# Inference Models ...
def uni_input_inference_model(LSTM_cell, densor, Ty=10):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """

    # Get the shape of input values
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units

    # Define the input of your model with a shape
    x0 = Input(shape=(1, n_values))

    # Define a0,c0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []

    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Perform one step of LSTM_cell. Use "x", not "x0" (≈1 line)
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])

        # Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)
        # Append the prediction "out" to "outputs". out.shape = (None, 90) (≈1 line)
        outputs.append(out)

        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = tf.math.argmax(out, -1)
        x = tf.one_hot(x, n_values)

        # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
        x = RepeatVector(1)(x)

    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


def multi_input_sequence_inference_model(LSTM_cell, densor, reshaper, Ty, Tx):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    tx -- integer, number of time steps for input
    Ty -- integer, number of time steps to generate
    sequence -- sequence of works you have done so far in that day

    Returns:
    inference_model -- Keras model instance
    """
    # Get the shape of input values
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units

    # Define the input of your model with a shape
    X = Input(shape=(Tx, n_values), name='X')  # (None, tx, n_values)

    # tx of current sequence in other hand number of works you have done so far in that day
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')  # init
    c0 = Input(shape=(n_a,), name='c0')  # init
    a = a0
    c = c0
    out = None
    # no need to save generated outputs we just need last a ,c and x
    for t in range(Tx):
        # Select the "t"th time step vector from X.
        x = X[:, t, :]  # (None, n_values)
        # Use reshaper to reshape x to be (1, n_values)
        x = reshaper(x)  # (None, 1, n_values)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)

    # So now we have new a, c
    x = tf.math.argmax(out, -1)
    x = tf.one_hot(x, n_values)
    # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
    x = RepeatVector(1)(x)

    # Create an empty list of "outputs" to later store your predicted values
    outputs = []
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)
        # Append the prediction "out" to "outputs". out.shape = (None, 90)
        outputs.append(out)
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        x = tf.math.argmax(out, -1)
        x = tf.one_hot(x, n_values)
        # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
        x = RepeatVector(1)(x)

    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[X, a0, c0], outputs=outputs)

    return inference_model


def multi_input_HiddenCell_sequence_inference_model(LSTM_cell, densor, reshaper, Ty, Tx):
    """
    multiInputHiddenCellSequenceInferenceModel() vs  multiInputSequenceInferenceModel() :
    is multiInputHiddenCellSequenceInferenceModel() against the multiInputSequenceInferenceModel()
    we can return hidden cells for firstSoftmax of beam search method

    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    tx -- integer, number of time steps for input
    Ty -- integer, number of time steps to generate
    sequence -- sequence of works you have done so far in that day

    Returns:
    inference_model -- Keras model instance
    """
    # Get the shape of input values
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units

    # Define the input of your model with a shape
    X = Input(shape=(Tx, n_values), name='X')  # (None, tx, n_values)

    # tx of current sequence in other hand number of works you have done so far in that day
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')  # init
    c0 = Input(shape=(n_a,), name='c0')  # init
    a = a0
    c = c0
    out = None
    # no need to save generated outputs we just need last a ,c and x
    for t in range(Tx):
        # Select the "t"th time step vector from X.
        x = X[:, t, :]  # (None, n_values)
        # Use reshaper to reshape x to be (1, n_values)
        x = reshaper(x)  # (None, 1, n_values)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)

    # So now we have new a, c
    x = tf.math.argmax(out, -1)
    x = tf.one_hot(x, n_values)
    # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
    x = RepeatVector(1)(x)

    # Create an empty list of "outputs" to later store your predicted values
    outputs = []
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)
        # Append the prediction "out" to "outputs". out.shape = (None, 90)
        outputs.append(out)
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        x = tf.math.argmax(out, -1)
        x = tf.one_hot(x, n_values)
        # RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
        x = RepeatVector(1)(x)

    # model instance with the correct "inputs" and "outputs"
    inference_model = Model(inputs=[X, a0, c0], outputs=[outputs, a, c])

    return inference_model