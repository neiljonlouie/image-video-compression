import os

import cv2
import keras
import numpy as np

class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=64):
        self.batch_size = batch_size

        if not os.path.isdir(root_dir):
            print('Error: %s is not a directory.' % root_dir)
            return

        self.image_dir = os.path.join(root_dir, 'data')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))
       
    def __len__(self):
        return np.int(np.ceil(len(self.label_filenames) / float(self.batch_size)))
        
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        x = []
        y = []

        for i in range(len(batch_x)):
            image_filename = os.path.join(self.image_dir, batch_x[i])
            label_filename = os.path.join(self.label_dir, batch_y[i])

            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)

            image_float = np.float32(image) / 255
            label_float = np.float32(label) / 255

            image_float = np.expand_dims(image_float, 2)
            label_float = np.expand_dims(label_float, 2)

            x.append(image_float)
            y.append(label_float)

        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
        
        return x_array, y_array