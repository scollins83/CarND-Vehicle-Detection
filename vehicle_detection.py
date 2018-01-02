import logging
import argparse
import json
import glob
import sys
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-c', '--configuration', help="File path configuration file", required=True,
                        dest='config')

    return vars(parser.parse_args(arguments))


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration


def get_image_file_paths(directory):
    """

    :param directory:
    :return: list of image file paths
    """
    image_types = os.listdir(directory)
    image_types = [imtype for imtype in image_types if not imtype.startswith('.')]

    image_list = []

    [image_list.extend(glob.glob(directory + imtype + '/*')) for imtype in image_types]

    return image_list


def get_hog_features(image, orient, pix_per_cell, cell_per_block,
                        visualize="False", feature_vec="True"):
    """
    Function copied from Udacity Self-Driving Car Nanodegree quiz.
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    # Convert string-based true or false values to boolean.
    if visualize == "True":
        visualize = True
    else:
        visualize = False

    if feature_vec == "True":
        feature_vec = True
    else:
        feature_vec = False

    if visualize == True:
        features, hog_image = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=visualize, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=visualize, feature_vector=feature_vec)
        return features


def bin_spatial(image, size=(32, 32)):
    """
    Spatially bins channels of a 3-channel image.
    :param image: Image array.
    :param size: Tuple, 2 values for image height and width for downsampling.
    :return: spatially binned color channels
    """
    assert image.shape[-1] == 3
    colors = []

    for channel in range(image.shape[-1]):
        colors.append(cv2.resize(image[:, :, channel], size).ravel())

    return np.hstack(tuple(colors))


def color_histogram(image, nbins=32):
    """

    :param image:
    :param nbins:
    :return:
    """
    assert image.shape[-1] == 3
    channel_hists = []

    for channel in range(image.shape[-1]):
        channel_hists.append(np.histogram(image[:, :, channel], bins=nbins)[0])

    # Return the concatenated histograms
    return np.concatenate(tuple(channel_hists))




if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger.info("Reading configuration file...")
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    logger.info("Reading image file lists...")
    cars = get_image_file_paths(config['vehicles_image_directory'])
    notcars = get_image_file_paths(config['nonvehicles_image_directory'])
    logger.info("Number of car images: " + str(len(cars)))
    logger.info("Number of non-car images: " + str(len(notcars)))

    sys.exit(0)
