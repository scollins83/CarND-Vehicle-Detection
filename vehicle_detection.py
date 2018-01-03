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


def convert_string_to_boolean(input_string):
    """

    :param input_string:
    :return: Boolean value
    """
    if input_string == "True":
        return True
    else:
        return False


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

    :param image:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param visualize:
    :param feature_vec:
    :return:
    """
    # Convert string-based true or false values to boolean.
    visualize = convert_string_to_boolean(visualize)
    feature_vec = convert_string_to_boolean(feature_vec)

    if visualize:
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


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat="True", hist_feat="True", hog_feat="True",
                     hog_visualize="False", hog_feature_vector="True"):
    """

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
    :param hog_visualize:
    :param hog_feature_vector:
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
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_histogram(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         visualize=hog_visualize,
                                                         feature_vec=hog_feature_vector))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, visualize=hog_visualize,
                                                feature_vec=hog_feature_vector)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """

    :param img:
    :param x_start_stop:
    :param y_start_stop:
    :param xy_window:
    :param xy_overlap:
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop is None:
        x_start_stop = [None, None]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(image, boxes, color=(0, 0, 255), thick=6):
    """

    :param image:
    :param boxes:
    :param color:
    :param thick:
    :return:
    """
    imcopy = np.copy(image)
    for box in boxes:
        cv2.rectangle(imcopy, box[0], box[1], color, thick)

    return imcopy

def extract_single_image_features(img, color_space='RGB', spatial_size=(32, 32),
                                  hist_bins=32, orient=9, pix_per_cell=8,
                                  cell_per_block=2, hog_channel=0,
                                  spatial_feat="True", hist_feat="True", hog_feat="True",
                                  hog_visualize="False", hog_feature_vector="True"):
    """

    :param img:
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

    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_histogram(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_visualize == "False":
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        visualize=hog_visualize, feature_vec=hog_feature_vector))
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, visualize=hog_visualize,
                                                feature_vec=hog_feature_vector)
            img_features.append(hog_features)

        else:
            if hog_channel == 'ALL':
                hog_features = []
                hog_images = []

                for channel in range(feature_image.shape[2]):
                    hog_feature, hog_image = get_hog_features(feature_image[:,:,channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              visualize=hog_visualize,
                                                              feature_vec=hog_feature_vector)
                    hog_features.append(hog_feature)
                    hog_images.append(hog_image)

                img_features.append(hog_features)
                return np.concatenate(img_features), hog_images


            else:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel],
                                                           orient, pix_per_cell,
                                                           cell_per_block,
                                                           visualize=hog_visualize,
                                                           feature_vec=hog_feature_vector)

                img_features.append(hog_features)
                return np.concatenate(img_features), hog_image


    #9) Return concatenated array of features
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    orient=9, pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat="True",
                    hist_feat="True", hog_feat="True"):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = extract_single_image_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def visualize(figure, rows, cols, imgs, titles):
    """

    :param figure:
    :param rows:
    :param cols:
    :param imgs:
    :param titles:
    :return:
    """
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


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

    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_features, car_hog_image = extract_single_image_features(car_image,
                                                                config['color_space'],
                                                                (config['image_height'],
                                                                 config['image_width']),
                                                                config['histogram_bins'],
                                                                config['orient'],
                                                                config['pix_per_cell'],
                                                                config['cell_per_block'],
                                                                config['hog_channel'],
                                                                config['extract_spatial_features'],
                                                                config['extract_histogram_features'],
                                                                config['extract_hog_features'],
                                                                config['visualize_hog'],
                                                                config['feature_vec_hog'])

    notcar_features, notcar_hog_image = extract_single_image_features(notcar_image,
                                                                      config['color_space'],
                                                                      (config['image_height'],
                                                                       config['image_width']),
                                                                      config['histogram_bins'],
                                                                      config['orient'],
                                                                      config['pix_per_cell'],
                                                                      config['cell_per_block'],
                                                                      config['hog_channel'],
                                                                      config['extract_spatial_features'],
                                                                      config['extract_histogram_features'],
                                                                      config['extract_hog_features'],
                                                                      config['visualize_hog'],
                                                                      config['feature_vec_hog'])

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
    fig = plt.figure(figsize=(12, 3))
    visualize(fig, 1, 4, images, titles)

    sys.exit(0)