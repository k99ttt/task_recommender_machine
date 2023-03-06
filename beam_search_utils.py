import numpy as np
from numpy import log
import tensorflow as tf


def index_to_model_input(indexes, n_values=293):
    oh = tf.one_hot(indexes, n_values, axis=-1)
    sw = np.swapaxes(oh, 0, 1)
    return sw


def current_value(matrix, width):
    fatten_matrix = matrix.flatten()
    value = fatten_matrix[np.sort((-matrix).flatten().argsort()[0:width])]
    return (np.expand_dims(value, axis=0)).T


def last_indexes(n_values, matrix, indexes, width):
    sort = (-matrix).flatten().argsort()[0:width]
    try:
        new_indexes = indexes[:, np.sort(sort) // n_values]
        new_index = np.sort(sort) % n_values
        new_index = np.expand_dims(new_index, axis=0)
        new_indexes = np.concatenate((new_indexes, new_index))
        return new_indexes
    except:
        return np.expand_dims((np.sort(sort) % n_values), axis=0)


def epsilon_function(epsilon_type):
    epsilon = 0
    if epsilon_type == '32':
        epsilon = np.finfo(np.float32).eps
    if epsilon_type == '64':
        epsilon = np.finfo(np.float64).eps
    return epsilon
