import os
import numpy as np
from graphic_utils import *
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # or 'INFO' instead of 'ERROR' for get Warnings and information's


def right_zeroes_remover(array):
    """array = [0,2,3,4,5,0,0,0,0,0,0,0,0] to array = [0,2,3,4,5] ;)"""
    array = np.array(array)
    x = np.where(array == 0)
    x = np.delete(x, 0)
    array = np.delete(array, x)
    array = list(array)
    return array


def model_loader(basePath='saved_models/models'):
    """load models from basepath(where do you start to reading ?) and write it to models array"""
    dirLenght = len(os.listdir(basePath)) + 1
    # don't worry about redundant length it will be removed by right_zeroes_remover() .
    models = [0] * dirLenght                   # init models array .
    # -------------------------------------------------------------------------------
    for entry in os.listdir(basePath):
        if entry != '.DS_Store':
            path = (basePath + entry)
            model = load_model(path)
            nOfInputs = model.input_shape[0][1]
            models[nOfInputs] = model
            print('model input_shape =>', model.input_shape)
            print('model output_shape =>', model.output_shape)
            print(path, f"loaded to => models[{nOfInputs}]")
            print(line + line)
            # -----------------------------------------------------------------------
    models = right_zeroes_remover(models)
    return models

# take a look at some examples ...
# model1Inputs.input_shape => [(None, 1, 8), (None, 32), (None, 32)]
# model1Inputs.output_shape => (None, 8)
# ------------------------------------------------------------------
# model2Inputs.input_shape => [(None, 2, 8), (None, 32), (None, 32)]
# model2Inputs.output_shape => (None, 8)
# ------------------------------------------------------------------
# model3Inputs.input_shape => [(None, 3, 8), (None, 32), (None, 32)]
# model3Inputs.output_shape => (None, 8)
# ------------------------------------------------------------------
# model4Inputs.input_shape => [(None, 4, 8), (None, 32), (None, 32)]
# model4Inputs.output_shape => (None, 8)
# ------------------------------------------------------------------
# now we got models includes multi input inference model with X inputs ...(X is 1 to maxLength)
