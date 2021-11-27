# This file converts the .mhd/.raw files in Classification_data_raw
# to 3-channel .png files to conform with what EfficientNet expects.
# It also does RandAugment with N=2, M=8.

import automl.autoaugment
import numpy as np
import skimage.io
import skimage.transform
import glob
import os.path
import imageio
import tensorflow as tf
import skimage.util
import skimage.color

def main():
    print('Creating files in original directory...')
    for path in glob.glob('Classification_data_raw/*/*/*.mhd', recursive=True):
        img = skimage.io.imread(path, plugin='simpleitk')
        filename = os.path.splitext(path)[0]
        if(np.shape(img) == (1, 256, 256)):
            img = np.squeeze(img, axis=0)
        img = skimage.transform.resize(img, (240, 240))
        img = skimage.color.gray2rgb(img)
        img = skimage.util.img_as_ubyte(img)
        if 'Training' in path:
            num_output = 10 if 'Healthy' in path else 5
            img_tensor = tf.convert_to_tensor(img)
            for i in range(num_output):
                new_img = automl.autoaugment.distort_image_with_randaugment(img_tensor, 2, 3)
                imageio.imwrite(filename+'_augment_'+str(i)+'.png', new_img.numpy(), format='png')
                print(filename+'_augment_'+str(i)+'.png')
        imageio.imwrite(filename+'.png', img, format='png')
        print(filename+'.png')


if __name__ == '__main__':
    main()