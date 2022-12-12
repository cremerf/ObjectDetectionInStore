from scripts.packages.paths import ODISPaths
import os

PATHS = ODISPaths()


data_folder = PATHS.DATA



# Instantiate the name of the folder's weights

training = 'first_training'

path_training = os.path.join(PATHS.WEIGHTS, training)



path_training_weights = os.path.join(path_training, 'weights')



name_of_weights = 'bestnoft.onnx'

path_picked_weights = os.path.join(path_training_weights, name_of_weights)

training = 'first_training'

path_training = os.path.join(PATHS.WEIGHTS, training)

path_training_weights = os.path.join(path_training, 'weights')

name_of_weights = 'bestnoft.onnx'

path_picked_weights = os.path.join(path_training_weights, name_of_weights)

yaml_file = 'config_blmodel.yaml'

path_yaml = os.path.join(PATHS.DATA, yaml_file)



print(PATHS.BASE_DIR)
print(PATHS.WEIGHTS)
print(path_training)
print(path_training_weights)
print(path_picked_weights)
print(path_yaml)




