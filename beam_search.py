import numpy as np
from numpy import log
import tensorflow as tf
from beam_search_utils import *


def beam_search(last_softmax, last_a, last_c, models, ty_max, epsilon_type='64', show=True, auto=True, beam_width=3):
    """
    Welcome to beam_search function ;) for more insight 'readMe/BeamSearch.md' is all you need
    """

    # epsilon for preventing from underflow
    epsilon = epsilon_function(epsilon_type)
    # initializing
    n_values = models[1].input_shape[0][-1]  # gets n_values directly from a model
    indexs = 0
    values = []
    model_matrix = None
    zero_index = None
    t = None
    # in step 0 we have 2 times adding index instead of 1 , so ty_max => ty_max - 1
    ty_max = ty_max - 1
    # ----------------------------------------------------------------
    for i in range(ty_max):
        # hyper parameter t for normalized or auto beam search
        t = (i + 1) ** 0.7
        # ------------------------------------------------------------
        if i == 0:
            model_matrix = last_softmax
            indexs = last_indexes(n_values, model_matrix, indexs, width=beam_width)
            zero_index = indexs.copy()  # just for printer
        # ------------------------------------------------------------
        value = current_value(model_matrix, width=beam_width)
        value_copy = value.copy()
        # ------------------------------------------------------------
        # select models[i+ 1] with i + 1 input ... to achieve model_matrix
        model_matrix = models[i + 1].predict([index_to_model_input(indexs, n_values), last_a, last_c])
        # modelMatrixCopy = model_matrix.copy() #just for printer
        if i == 0:
            # + epsilon for preventing -inf and in general log() is for preventing memory underflow
            value = log(
                value + epsilon)
        model_matrix = value + log(
            model_matrix + epsilon)

        values.append((value / t).tolist())
        # ------------------------------------------------------------
        indexs = last_indexes(n_values, model_matrix, indexs, width=beam_width)
        # ---------------------------Printer--------------------------
        if show:
            print('------------- step ' + str(i) + ' -------------')
            if i == 0:
                print('firstSoftmax.shape => \n', last_softmax.shape)
                print('firstSoftmax => \n', last_softmax)
                print('zero_index => \n', zero_index)
            print('number of model inputs =>', i + 1)
            print('value_copy => \n', value_copy)
            print('model_matrix => \n', model_matrix)
            print('model_matrix.shape => \n', model_matrix.shape)
            # print('value => \n',value)
            print('indexes => \n', indexs)
    # ------------------------------------------------------------
    value = current_value(model_matrix, width=beam_width)
    if show:
        print('finalValue => \n', value)
    # ---------------------------------------------
    values.append((value / t).tolist())
    values = np.array(values)
    # -------- normalized beam search -------------
    swapped_values = np.swapaxes(values.copy(), 0, 2)
    values_len = swapped_values.shape[-1]
    row = values.argmax() // values_len
    col = values.argmax() % values_len
    auto_beam_sequence = indexs.T[row, :col + 1]
    # ---------------------------------------------
    beam_sequence = np.array(indexs)[:, value.argmax()]
    # ---------------------------------------------
    if auto:
        return auto_beam_sequence
    else:
        return beam_sequence

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Built with Love For Chargoon Company -----------------------------------------
# --------------------------------------- Kasra , Mohammad Javad and Shahbaz -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
