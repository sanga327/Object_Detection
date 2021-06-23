# load data
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels

# %matplotlib inline




# make h5 file
# 데이터set을 만드는 부분
from data_generator.object_detection_2d_data_generator import DataGenerator

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets.

# TODO: Set the paths to your dataset here.

# Images
images_dir = 'data/'

# Ground truth
train_labels_filename = 'data/train/train.csv'
val_labels_filename = 'data/val/val.csv'
test_labels_filename ='data/test/test.csv'

train_dataset.parse_csv(images_dir=images_dir+'train/',
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir+'val/',
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

test_dataset.parse_csv(images_dir=images_dir+'test/',
                      labels_filename=test_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

train_dataset.create_hdf5_dataset(file_path='saved_model/dataset_train.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='saved_model/dataset_val.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

test_dataset.create_hdf5_dataset(file_path='saved_model/dataset_test.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()
test_dataset_size = test_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
print("Number of images in the test dataset:\t{:>6}".format(test_dataset_size))

# train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='saved_model/dataset_train.h5')
# val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='saved_model/dataset_val.h5')
# test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='saved_model/dataset_test.h5')
#
# train_dataset_size = train_dataset.get_dataset_size()
# val_dataset_size   = val_dataset.get_dataset_size()
# test_dataset_size   = test_dataset.get_dataset_size()
#
# print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
# print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
# print("Number of images in the validation dataset:\t{:>6}".format(test_dataset_size))



