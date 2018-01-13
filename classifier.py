import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
import sys
import logging
import pickle
import argparse
import json
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog


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


def convert_string_to_boolean(input_string):
    """

    :param input_string:
    :return: Boolean value
    """
    if input_string == "True":
        return True
    else:
        return False


def bin_spatial(img, size=(32, 32)):
    """
    Define a function to compute binned color features
    :param img:
    :param size:
    :return:
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color histogram features
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Define a function to extract features from a list of images
    Have this function call bin_spatial() and color_hist()

    :param imgs:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """

    spatial_feat = convert_string_to_boolean(spatial_feat)
    hist_feat = convert_string_to_boolean(hist_feat)
    hog_feat = convert_string_to_boolean(hog_feat)
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


if __name__ == "__main__":

    logger.info("Reading configuration file...")

    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Read in car and non-car images
    #cars = glob.glob(config['vehicles_image_directory'] + '/*.jpg')
    #notcars = glob.glob(config['nonvehicles_image_directory'] + '/*.jpg')

    cars = get_image_file_paths(config['vehicles_image_directory'])
    notcars = get_image_file_paths(config['nonvehicles_image_directory'])

    logger.info(len(cars))
    logger.info(len(notcars))

    car_features = extract_features(cars,
                                    color_space=config['color_space'],
                                    spatial_size=(config['image_height'],
                                                  config['image_width']),
                                    hist_bins=config['histogram_bins'],
                                    orient=config['orient'],
                                    pix_per_cell=config['pix_per_cell'],
                                    cell_per_block=config['cell_per_block'],
                                    hog_channel=config['hog_channel'],
                                    spatial_feat=config['extract_spatial_features'],
                                    hist_feat=config['extract_histogram_features'],
                                    hog_feat=config['extract_hog_features'])

    notcar_features = extract_features(notcars,
                                       color_space=config['color_space'],
                                       spatial_size=(config['image_height'],
                                                     config['image_width']),
                                       hist_bins=config['histogram_bins'],
                                       orient=config['orient'],
                                       pix_per_cell=config['pix_per_cell'],
                                       cell_per_block=config['cell_per_block'],
                                       hog_channel=config['hog_channel'],
                                       spatial_feat=config['extract_spatial_features'],
                                       hist_feat=config['extract_histogram_features'],
                                       hog_feat=config['extract_hog_features'])

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    logger.info('Using spatial binning of: ' + str(config['image_height']) +
          ' and ' + str(config['histogram_bins']) + ' histogram bins')
    logger.info('Feature vector length: ' + str(len(X_train[0])))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    logger.info(str(round(t2 - t, 2)) + ' seconds to train SVC...')

    # Check the score of the SVC
    logger.info('Test Accuracy of SVC = ' + str(round(svc.score(X_test, y_test), 4)))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    with open(config['classifier_file'], 'wb') as pkl_file:
        pickle.dump(svc, pkl_file)

    sys.exit(0)
