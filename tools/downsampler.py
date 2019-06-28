# Usage python3 downsampler.py /path/to/input /path/to/output

import sys
import numpy as np

import cv2

input_filename = sys.argv[1]
output_filename = sys.argv[2]

# TODO: Provide these as parameters
width = 1920
height = 1080
scale = 4
frame_size = int(width * height * 1.5)

input_file = open(input_filename, 'rb')
output_file = open(output_filename, 'wb')

bytes = input_file.read(frame_size)
while len(bytes) == frame_size:
    yuv = np.frombuffer(bytes, dtype=np.uint8)
    yuv = yuv.reshape((int(height * 1.5), width))
    yuv_resized = cv2.resize(yuv, dsize=(int(width / scale), int(height * 1.5 / scale)))
    output_file.write(yuv_resized)

    # rgb = cv2.cvtColor(yuv_resized, cv2.COLOR_YUV2RGB_YV12)
    # cv2.namedWindow('frame')
    # cv2.imshow('frame', rgb)
    # cv2.waitKey(15)

    bytes = input_file.read(frame_size)

input_file.close()
output_file.close()