import numpy as np
from beam_search_utils import *
from beam_search import *


def beam_search_test(models, n_values):
    xInit = np.random.rand(1, 1, n_values)         # (m,tx,n_values) (1, 1, 293)
    aInit = np.zeros((1, 32))                      # (1,n) (1, 32)
    cInit = np.zeros((1, 32))                      # (1,n) (1, 32)
    # ----------------------------------------------------------------
    # in this part we assume that we feed our X's to our model that's time for generating
    # Function parameters
    Ty_max = 6
    lastSoftmax = models[1].predict([xInit, aInit, cInit])
    lastA = np.zeros((3, 32))                      # (beam_width,n) (3, 32) processed hidden state from last model
    lastC = np.zeros((3, 32))                      # (beam_width,n) (3, 32) processed hidden cell from last model
    # ----------------------------------------------------------------
    sequence = beam_search(lastSoftmax, lastA, lastC, models, Ty_max, epsilon_type='32', show=True, auto=False)
    return sequence


def beam_search_test_show():
    """
    no teed to last_softmax,Ty ,show , last_a, last_c, models and
    epsilon_type for input in test(beam_search_test_show)...
    for more information please read => readMe/BeamSearch.md
    """
    # test parameters
    n_values = 5
    Ty = 6
    show = True
    # --------------------------------------
    np.random.seed(seed=2)
    # l1------------------------------------
    a = np.random.rand(n_values).round(1)
    # l2------------------------------------
    a10 = np.random.rand(n_values).round(1)
    a20 = np.random.rand(n_values).round(1)
    a30 = np.random.rand(n_values).round(1)
    # l3------------------------------------
    a11 = np.random.rand(n_values).round(1)
    a21 = np.random.rand(n_values).round(1)
    a31 = np.random.rand(n_values).round(1)
    # l4------------------------------------
    a12 = np.random.rand(n_values).round(1)
    a22 = np.random.rand(n_values).round(1)
    a32 = np.random.rand(n_values).round(1)
    # l5------------------------------------
    a13 = np.random.rand(n_values).round(1)
    a23 = np.random.rand(n_values).round(1)
    a33 = np.random.rand(n_values).round(1)
    # l6------------------------------------
    a14 = np.random.rand(n_values).round(1)
    a24 = np.random.rand(n_values).round(1)
    a34 = np.random.rand(n_values).round(1)
    # l7-------------------------------------
    a15 = np.random.rand(n_values).round(1)
    a25 = np.random.rand(n_values).round(1)
    a35 = np.random.rand(n_values).round(1)
    # l7-------------------------------------
    a16 = np.random.rand(n_values).round(1)
    a26 = np.random.rand(n_values).round(1)
    a36 = np.random.rand(n_values).round(1)
    # ---------------------------------------
    mat0 = np.stack((a10, a20, a30))
    mat1 = np.stack((a11, a21, a31))
    mat2 = np.stack((a12, a22, a32))
    mat3 = np.stack((a13, a23, a33))
    mat4 = np.stack((a14, a24, a34))
    mat5 = np.stack((a15, a25, a35))
    mat6 = np.stack((a16, a26, a36))
    mat = np.stack((mat0, mat1, mat2, mat3, mat4, mat5, mat6))
    # ----------------------------------------------------------------
    # in this part we assume that we feed our X's to our model and thats time for generating
    # Function parameters
    # ----------------------------------------------------------------
    # lastA = None  # no need to use this in test
    # lastC = None  # no need to use this in test
    # ----------------------------------------------------------------
    # epsilon for preventing from underflow
    # initializing indexes
    indexs = 0
    values = []
    lastSoftmax = a
    modelMatrix = None
    zeroIndex = None
    t = None
    # ----------------------------------------------------------------
    for i in range(Ty):
        t = (i + 1) ** 0.7
        if i == 0:
            modelMatrix = lastSoftmax
            indexs = last_indexes(n_values, modelMatrix, indexs, width=3)
            zeroIndex = indexs.copy()  # just for printer
        # ------------------------------------------------------------
        value = current_value(modelMatrix, width=3)
        valueCopy = value.copy()
        values.append((value/t).tolist())
        # ------------------------------------------------------------
        modelMatrix = mat[i]
        # ------------------------------------------------------------
        # modelMatrixCopy = modelMatrix.copy() #just for printer
        if i == 0:
            pass
        modelMatrix = value + modelMatrix
        # ------------------------------------------------------------
        indexs = last_indexes(n_values, modelMatrix, indexs, width=3)
        # ---------------------------Printer--------------------------
        if show:
            print('------------- step '+str(i)+' -------------')
            if i == 0:
                print('firstSoftmax.shape => \n', lastSoftmax.shape)
                print('firstSoftmax => \n', lastSoftmax)
                print('zero_index => \n', zeroIndex)
            print('number of model inputs =>', i+1)
            print('valueCopy => \n', valueCopy)
            print('model_matrix => \n', modelMatrix)
            print('model_matrix.shape => \n', modelMatrix.shape)
            # print('value => \n',value)
            print('indexes => \n', indexs)
        # ------------------------------------------------------------
    value = current_value(modelMatrix, width=3)
    # ---------------------------------------------
    # valueCopy = value.copy()
    values.append((value/t).tolist())
    values = np.array(values)
    # -------- normalized beam search -------------
    swapedValues = np.swapaxes(values.copy(), 0, 2)
    valuesLen = swapedValues.shape[-1]
    # beam_width = swapedValues.shape[1]
    row = values.argmax() // valuesLen
    col = values.argmax() % valuesLen
    # swapedValues = np.squeeze(swapedValues, axis=0)
    autoBeamSequence = indexs.T[row, :col + 1]
    # ---------------------------------------------
    if show:
        print('values=> \n', values)
        print('finalValue => \n', value)
        print(autoBeamSequence)
    return autoBeamSequence


beam_search_test_show()
