import argparse, os
from time import time

import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

import model
from generator import DatasetGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Settings
image_size = (64, 64, 1)
batch_size = 32
epochs = 20
train_root_dir = '/home/neil/Datasets/DIV2K/train-patched'
valid_root_dir = '/home/neil/Datasets/DIV2K/valid-patched'

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights_file', help='Weights file (.hdf5) to continue training')
parser.add_argument('--num_epochs', type=int, help='Number of epochs to be run during training')
args = parser.parse_args()

# Train the model
train_generator = DatasetGenerator(train_root_dir, batch_size)
valid_generator = DatasetGenerator(valid_root_dir, batch_size)

model = model.build_model(image_size)
model.summary()
if not args.weights_file is None:
    model.load_weights(args.weights_file)

if not args.num_epochs is None:
    epochs = args.num_epochs

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

filepath = 'weights-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=2)

adam = Adam(lr=0.002)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
model.fit_generator(train_generator, epochs=epochs, verbose=1, validation_data=valid_generator,
                    callbacks=[early_stop, tensorboard, checkpoint])
