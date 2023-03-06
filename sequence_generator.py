from sequence_generator_utils import sequence_generator
from load_model import *
from data_analyst import *

# learning_type : for use learned weight from small_model => set to 'small_model'
# for use learned weight from model => set to 'model'

learning_type = 'small_model'

if learning_type == 'model':
    print('model selected as learned model ...')
    models = model_loader('saved_models/models/')
    hiddenCellModels = model_loader('saved_models/hc_models/')

elif learning_type == 'small_model':
    print('small_model selected as learned model ...')
    models = model_loader('saved_models/small_models/')
    hiddenCellModels = model_loader('saved_models/hc_small_models/')

else:
    models = None
    hiddenCellModels = None


currentSequence = np.ones((1, 4, 293)) * 0  # what have you done so far ...
ourSequence = sequence_generator(currentSequence, hiddenCellModels, models, ty_max=10, beam_width=4, auto=True)


print('ourSequence => ', ourSequence)
array_plotter(ourSequence)


# it seams right ...


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Built with Love For Chargoon Company -----------------------------------------
# --------------------------------------- Kasra , Mohammad Javad and Shahbaz -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
