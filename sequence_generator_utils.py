import numpy as np
from beam_search import beam_search


def copy_and_stack(array, beam_width):
    matrix = np.zeros((beam_width, array.shape[-1]))
    arange = np.arange(0, beam_width)
    matrix[arange, :] = array.copy()
    return matrix


def sequence_generator(current_sequence, hidden_cell_models, models, ty_max=8, show=False, auto='auto', beam_width=3,
                       n_a=64):

    init_a = np.zeros((1, n_a))  # init values (m,n_values) (1, 32)
    init_c = np.zeros((1, n_a))  # init values (m,n_values) (1, 32)

    print('sequence_generator ...')
    number_of_inputs = current_sequence.shape[1]
    model = hidden_cell_models[number_of_inputs]
    last_softmax, last_a, last_c = model.predict([current_sequence, init_a, init_c])

    print('last_a.shape =>', last_a.shape)
    print('last_c.shape =>', last_c.shape)
    print('last_softmax.shape =>', np.array(last_softmax).shape)
    last_a = copy_and_stack(last_a, beam_width)
    last_c = copy_and_stack(last_c, beam_width)
    last_softmax = np.squeeze(np.array(last_softmax), axis=0)

    sequence = beam_search(last_softmax, last_a, last_c, models, ty_max=ty_max, epsilon_type='64', show=show, auto=auto,
                           beam_width=beam_width)

    return sequence
