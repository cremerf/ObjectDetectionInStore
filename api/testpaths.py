from paths import ODISPaths
import os

PATHS = ODISPaths()

print(PATHS.BASE_DIR)

data_folder = PATHS.DATA

print(data_folder)

print(PATHS.WEIGHTS)

# Instantiate the name of the folder's weights

training = 'first_training'

path_training = os.path.join(PATHS.WEIGHTS, training)

print(path_training)

path_training_weights = os.path.join(path_training, 'weights')

print(path_training_weights)

name_of_weights = 'bestnoft.onnx'

path_picked_weights = os.path.join(path_training_weights, name_of_weights)

print(path_picked_weights)



