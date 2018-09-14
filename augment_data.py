import re
import random
import numpy as np
from scipy.misc import imread, imsave, imresize
import os.path
from glob import glob
from imgaug import augmenters as iaa

def augment_data(image, gt_image):
    """
    Apply augmentation techniques to a given image.
    :param images: Input image
    :param gt_images: Corresponding gt image
    """
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to all or some images
    seq1 = iaa.Sequential(
        [
            iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 20px (randomly chosen)
            iaa.Fliplr(1.0), # always horizontally flip each input image
            sometimes(iaa.GaussianBlur(sigma=(0, 2.0))), # blur images with a sigma of 0 to 2.0
            iaa.Affine(rotate=(-45, 45)), # rotate by -45 to +45 degrees]
        ]
    )

    seq2 = iaa.Sequential(
        [
            iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 20px (randomly chosen)
            iaa.Flipud(1.0), # always vertically flip each input image
            sometimes(iaa.GaussianBlur(sigma=(0, 2.0))), # blur images with a sigma of 0 to 2.0
            iaa.Affine(rotate=(-45, 45)), # rotate by -45 to +45 degrees
        ]
    )

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq1_det = seq1.to_deterministic()
    seq2_det = seq2.to_deterministic()

    image_aug_1 = seq1_det.augment_image(image)
    image_aug_2 = seq2_det.augment_image(image)
    gt_image_aug_1 = seq1_det.augment_image(gt_image)
    gt_image_aug_2 = seq2_det.augment_image(gt_image)

    return np.array([image_aug_1, image_aug_2]), np.array([gt_image_aug_1, gt_image_aug_2])

def read_and_augment_data(data_folder, image_shape):
    # Retrieve paths for all images and gt images
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    # Read each image and the corresponding gt image
    for image_file in image_paths:
        image_file_basename = os.path.basename(image_file)
        gt_image_file = label_paths[image_file_basename]
        gt_image_file_basename = os.path.basename(gt_image_file)

        print('Processing {} and {}'.format(image_file_basename, gt_image_file_basename))

        image = imresize(imread(image_file), image_shape)
        gt_image = imresize(imread(gt_image_file), image_shape)

        # Augment image and gt image
        images_aug, gt_images_aug = augment_data(image, gt_image)

        # Saves augmented images
        for i, (image_aug, gt_image_aug) in enumerate(zip(images_aug, gt_images_aug)):
            image_filename = os.path.splitext(image_file_basename)[0] + '_aug_' + str(i) + '.png'
            gt_image_filename = os.path.splitext(gt_image_file_basename)[0] + '_aug_' + str(i) + '.png'
            imsave(os.path.join(data_folder, 'image_2', image_filename), image_aug.astype(np.float32))
            imsave(os.path.join(data_folder, 'gt_image_2', gt_image_filename), gt_image_aug.astype(np.float32))

def run():
    data_folder = os.path.join('./data', 'data_road/training')
    image_shape = (375, 1242)

    read_and_augment_data(data_folder, image_shape)

    print('Done.')

if __name__ == '__main__':
    run()
