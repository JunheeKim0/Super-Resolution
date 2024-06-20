from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import argparse

from data_generator import *

DATA_DIR = "D:/HCP dataset/process/T1_pre"


# total_train = get_data(DATA_DIR, 2)

training_hr, training_lr = get_hr_lr(DATA_DIR)


# training_generator = train_data_generator_3d(total_train[0], total_train[1], 1)

# training_lr, training_hr = next(training_generator)

print(training_lr.shape, training_hr.shape)


# with open(output_path + f'/IXI_training_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
#     np.save(f, training_hr)

