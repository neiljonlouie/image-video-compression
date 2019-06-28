# Usage: python3 generate_images.py /path/to/input /path/to/output

import os, random, sys
import cv2

input_dir = sys.argv[1]
output_dir = sys.argv[2]

input_lr = os.path.join(input_dir, 'lr')
input_hr = os.path.join(input_dir, 'hr')
output_lr = os.path.join(output_dir, 'data')
output_hr = os.path.join(output_dir, 'label')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(output_hr):
    os.mkdir(output_hr)
if not os.path.exists(output_lr):
    os.mkdir(output_lr)

file_list = sorted(os.listdir(input_hr))
for filename in file_list:
    print('Generating patches for %s...' % filename)

    parts = filename.split('.')
    hr_path = os.path.join(input_hr, filename)
    lr_path = os.path.join(input_lr, parts[0] + 'x4.' + parts[1])

    hr = cv2.imread(hr_path)
    lr = cv2.imread(lr_path)

    hr_yuv = cv2.cvtColor(hr, cv2.COLOR_RGB2YUV)
    lr_yuv = cv2.cvtColor(lr, cv2.COLOR_RGB2YUV)

    hr_y = cv2.extractChannel(hr_yuv, 0)
    lr_y = cv2.extractChannel(lr_yuv, 0)

    hr_shape = hr_y.shape
    lr_shape = lr_y.shape

    num_patches = 0
    for i in range(0, lr_shape[0] - 64, 32):
        for j in range(0, lr_shape[1] - 64, 32):
            hr_patch = hr_y[4*i:4*i+256, 4*j:4*j+256]
            lr_patch = lr_y[i:i+64, j:j+64]

            num_patches += 1
            hr_patch_filename = '%s_%04d.png' % (parts[0], num_patches)
            lr_patch_filename = '%s_%04d.png' % (parts[0], num_patches)
            cv2.imwrite(os.path.join(output_hr, hr_patch_filename), hr_patch)
            cv2.imwrite(os.path.join(output_lr, lr_patch_filename), lr_patch)

            flip = random.randint(-1, 1)
            hr_patch = cv2.flip(hr_patch, flip)
            lr_patch = cv2.flip(lr_patch, flip)

            num_patches += 1
            hr_patch_filename = '%s_%04d.png' % (parts[0], num_patches)
            lr_patch_filename = '%s_%04d.png' % (parts[0], num_patches)
            cv2.imwrite(os.path.join(output_hr, hr_patch_filename), hr_patch)
            cv2.imwrite(os.path.join(output_lr, lr_patch_filename), lr_patch)

print('Done.')