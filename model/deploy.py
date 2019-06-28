import argparse, os

import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Settings
image_height = 270
image_width = 480
scale = 4
patch_size = (64, 64, 1)
frame_size = int(image_height * image_width * 1.5)

final_height = image_height * scale
final_width = image_width * scale
final_patch_size = (patch_size[0] * scale, patch_size[1] * scale, 1)


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights_file', help='Weights file (.hdf5) to be used for training or testing')
parser.add_argument('--input_file', help='Path to raw YUV file')
parser.add_argument('--output_file', help='Path to store the super-resized images')
args = parser.parse_args()

# Deploy the trained model
if args.weights_file is None:
    print('Error: Weights file needed for prediction. Please provide a weights file using the --weights_file argument.')
    quit()
    
if args.input_file is None:
    print('Error: Input file needed for prediction. Please provide an input file using the --input_file argument.')
    quit()

if not os.path.exists(args.input_file):
    print ('File %s not found.' % args.input_file)
    quit()

model = model.build_model(patch_size)
model.load_weights(args.weights_file)

input_file = open(args.input_file, 'rb')
output_file = open(args.output_file, 'wb')

bytes = input_file.read(frame_size)
num_processed = 0
while len(bytes) == frame_size:
    yuv = np.frombuffer(bytes, dtype=np.uint8)
    yuv = yuv.reshape((int(image_height * 1.5), image_width))
    
    patches = []
    for i in range(0, image_height - 1, patch_size[0]):
        for j in range(0, image_width - 1, patch_size[1]):
            patch = np.zeros(shape=(patch_size[0], patch_size[1]))

            if i + patch_size[0] <= image_height:
                if j + patch_size[1] <= image_width:
                    patch = yuv[i:i+patch_size[0], j:j+patch_size[1]]
                else:
                    remaining = image_width - j
                    patch[:, 0:remaining] = yuv[i:i+patch_size[0], j:j+remaining]
            else:
                remaining_i = image_height - i
                if j + patch_size[1] <= image_width:
                    patch[0:remaining_i, :] = yuv[i:i+remaining_i, j:j+patch_size[1]]
                else:
                    remaining_j = image_width - j
                    patch[0:remaining_i, 0:remaining_j] = yuv[i:i+remaining_i, j:j+remaining_j]

            patch = np.float32(patch) / 255
            patch = np.expand_dims(patch, 2)
            patches.append(patch)

    patches = np.array(patches)
    
    pred = model.predict(patches, batch_size=10, verbose=1)
    pred = np.squeeze(pred)
    pred = np.uint8(pred * 255)

    counter = 0
    yuv_resized = np.zeros(shape=(int(image_height * scale * 1.5), image_width * scale),
                           dtype=np.uint8)

    for i in range(0, final_height - 1, final_patch_size[0]):
        for j in range(0, final_width - 1, final_patch_size[1]):
            if i + final_patch_size[0] <= final_height:
                if j + final_patch_size[1] <= final_width:
                    yuv_resized[i:i+final_patch_size[0], j:j+final_patch_size[1]] = \
                        pred[counter, :, :]
                else:
                    remaining = final_width - j
                    yuv_resized[i:i+final_patch_size[0], j:j+remaining] = \
                        pred[counter, :, 0:remaining]
            else:
                remaining_i = final_height - i
                if j + final_patch_size[1] <= final_width:
                    yuv_resized[i:i+remaining_i, j:j+final_patch_size[1]] = \
                        pred[counter, 0:remaining_i, :]
                else:
                    remaining_j = final_width - j
                    yuv_resized[i:i+remaining_i, j:j+remaining_j] = \
                        pred[counter, 0:remaining_i, 0:remaining_j]

            counter += 1

    yuv_resized[final_height:, :] = cv2.resize(yuv[image_height:, :],
                                        dsize=(int(final_width), int(final_height / scale * 2)))
    output_file.write(yuv_resized)
    
    # rgb = cv2.cvtColor(yuv_resized, cv2.COLOR_YUV2RGB_YV12)
    # cv2.namedWindow('frame')
    # cv2.imshow('frame', rgb)
    # cv2.waitKey(15)

    num_processed += 1
    print('Done processing %d frame(s).' % num_processed)
    bytes = input_file.read(frame_size)

input_file.close()
output_file.close()