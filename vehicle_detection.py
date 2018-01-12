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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle

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


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    """

    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
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

    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
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
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_histogram(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_visualize == "False":
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         visualize=hog_visualize, feature_vec=hog_feature_vector))
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, visualize=hog_visualize,
                                                feature_vec=hog_feature_vector)
            img_features.append(hog_features)

        else:
            if hog_channel == 'ALL':
                hog_features = []
                hog_images = []

                for channel in range(feature_image.shape[2]):
                    hog_feature, hog_image = get_hog_features(feature_image[:, :, channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              visualize=hog_visualize,
                                                              feature_vec=hog_feature_vector)
                    hog_features.append(hog_feature)
                    hog_images.append(hog_image)

                img_features.append(hog_features)
                return np.concatenate(img_features), hog_images


            else:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel],
                                                           orient, pix_per_cell,
                                                           cell_per_block,
                                                           visualize=hog_visualize,
                                                           feature_vec=hog_feature_vector)

                img_features.append(hog_features)
                return np.concatenate(img_features), hog_image

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat="True",
                   hist_feat="True", hog_feat="True"):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = extract_single_image_features(test_img, color_space=color_space,
                                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                                 orient=orient, pix_per_cell=pix_per_cell,
                                                 cell_per_block=cell_per_block,
                                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                 hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def visualize(figure, rows, cols, imgs, titles, filename):
    """

    :param figure:
    :param rows:
    :param cols:
    :param imgs:
    :param titles:
    :return:
    """
    # TODO: Modify to capture runtime in filename.
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.title(i + 1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
            plt.savefig(filename, format='png')
        else:
            plt.imshow(img)
            plt.title(titles[i])
            plt.savefig(filename, format='png')


def convert_color(image, conv='RGB2YCrCb'):
    """
    Converts color space.
    :param image: Input image
    :param conv:
    :return:
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif conv == 'RGB2LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)


def find_cars(img, scale, y_start, y_stop, orient, pix_per_cell, cell_per_block, window, image_height,
              image_width, histogram_bins, red, green, blue, X_scaler):
    """

    :param img:
    :param scale:
    :return:
    """
    count = 0
    # Copy image to draw on
    draw_img = np.copy(img)

    # Make Heatmap
    heatmap = np.zeros_like((img[:, :, 0]))

    # PNG to JPG adjustment
    img = img.astype(np.float32) / 255.  # Needed if trained on png, but are now using jpg
    logger.info(str(np.min(img)) + ',' + str(np.max(img)))

    # Crop the image
    img_tosearch = img[y_start:y_stop, :, :]
    color_translation_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1.0:
        imshape = color_translation_tosearch.shape
        color_translation_tosearch = cv2.resize(color_translation_tosearch,
                                                (np.int(imshape[1]/scale),
                                                 np.int(imshape[0]/scale)))
    channel_1 = color_translation_tosearch[:, :, 0]
    channel_2 = color_translation_tosearch[:, :, 1]
    channel_3 = color_translation_tosearch[:, :, 2]

    # Define blocks and steps (note: might be duplicate of sliding_window function)
    nx_blocks = (channel_1.shape[1] // pix_per_cell) - 1
    ny_blocks = (channel_1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2

    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (ny_blocks - nblocks_per_window) // cells_per_step + 1
    hog1 = get_hog_features(channel_1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(channel_2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(channel_3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(color_translation_tosearch[ytop:ytop + window, xleft:xleft + window],
                                (window, window))

            # Get color features
            spatial_features = bin_spatial(subimg, size=(image_height, image_width))
            hist_features = color_hist(subimg, nbins=histogram_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start),
                              (xbox_left + win_draw, ytop_draw + win_draw + y_start),
                              (red, green, blue))
                img_boxes.append(((xbox_left, ytop_draw + y_start),
                                  xbox_left + win_draw, ytop_draw + win_draw + y_start))
                heatmap[ytop_draw + y_start:ytop_draw + win_draw + y_start,
                xbox_left:xbox_left + win_draw] += 1

    return draw_img, heatmap, img_boxes, count


def apply_threshold(heatmap, threshold):
    """
    Zeros out pixels in the heatmap with values below the specified threshold.
    :param heatmap:
    :param threshold:
    :return:
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    """

    :param img:
    :param labels:
    :return:
    """
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car number label value
        nonzero = (labels[0] == car_number).nonzero()

        #ID x and y values of nonzero pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def process_image(img):
    """

    :param img:
    :return:
    """
    out_img, heat_map, _, _ = find_cars(img, config['scale'], config['y_start'], config['y_stop'],
                                  config['orient'], config['pix_per_cell'], config['cell_per_block'],
                                  config['window'], config['image_height'], config['image_width'],
                                  config['histogram_bins'], config['box_color_red'],
                                  config['box_color_green'], config['box_color_blue'], X_scaler)
    heat_map = apply_threshold(heat_map, 2)
    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


def preprocess_for_training():
    t = time.time()
    config = load_config('configurations/local_test_configuration.json')

    logger.info("Reading image file lists...")
    cars = get_image_file_paths(config['vehicles_image_directory'])
    notcars = get_image_file_paths(config['nonvehicles_image_directory'])

    if len(cars) == 0 and len(notcars) == 0:
        cars = glob.glob(config['vehicles_image_directory'] + '/*.jpg')
        notcars = glob.glob(config['nonvehicles_image_directory'] + '/*.jpg')

    logger.info(len(cars))
    logger.info(len(notcars))
    random_idxs = np.random.randint(0, len(cars), config['n_samples'])
    if len(cars) <= config['n_samples']:
        test_cars = cars
        test_notcars = notcars
    else:
        test_cars = [cars[i] for i in random_idxs]
        test_notcars = [cars[i] for i in random_idxs]
    car_features = extract_features(test_cars,
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

    notcar_features = extract_features(test_notcars,
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

    seconds = np.round((time.time() - t), 4)
    logger.info('Seconds to compute features... ' + str(seconds))
    logger.info(len(car_features))
    logger.info(len(notcar_features))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Scale X features
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    logger.info('X: ' + str(len(scaled_X)))

    # Define labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    logger.info('y: ' + str(len(y)))

    ##  Test/Train Sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=config['test_set_percent'],
                                                        shuffle=True)
    logger.info('Using: ' + str(config['orient']) + ' orientations, ' +
                str(config['pix_per_cell']) + ' pixels per cell, ' +
                str(config['cell_per_block']) + ' cells per block, ' +
                str(config['histogram_bins']) + ' histogram bins, and (' +
                str(config['image_height']) + ',' + str(config['image_width']) + ') spatial sampling.')
    logger.info('Feature vector length: ' + str(len(X_train[0])))

    return X_scaler, X_train, X_test, y_train, y_test


def train_svc(X, y, X_test, y_test, C=5):
    global svc, t
    # Using a linear SVC
    # parameters = {'C': [1, 5, 10]}
    svc = LinearSVC(C=C)
    t = time.time()
    # clf = GridSearchCV(svr, parameters)
    svc.fit(X, y)
    logger.info(str(round(time.time() - t, 2)) + ' seconds to train SVC...')
    # best = clf.best_estimator_
    logger.info(svc)
    # Check SVC score
    logger.info('Test accuracy of Classifier = ' + str(svc.score(X_test, y_test)))

    return svc


if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger.info("Reading configuration file...")
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # logger.info("Number of car images: " + str(len(cars)))
    # logger.info("Number of non-car images: " + str(len(notcars)))

    if config['single_image_display'] == "True":

        logger.info("Reading image file lists...")
        cars = get_image_file_paths(config['vehicles_image_directory'])
        notcars = get_image_file_paths(config['nonvehicles_image_directory'])

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
        logger.info(car_features.shape)
        logger.info(notcar_features.shape)

        images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
        titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
        fig = plt.figure(figsize=(10, 10))
        visualize(fig, 2, 2, images, titles, config['hog_vis_filename'])


    t = time.time()
    X_scaler, X_train, X_test, y_train, y_test = preprocess_for_training()

    if config['train_classifier'] == "True":
        svc = train_svc(X_train, y_train, X_test, y_test)
        with open(config['classifier_file'], 'wb') as pkl_file:
            pickle.dump(svc, pkl_file)
    else:
        with open(config['classifier_file'], 'rb') as pkl_file:
            svc = pickle.load(pkl_file)

    # Start with video
    out_images = []
    out_titles = []

    y_start_stop = [config['y_start'], config['y_stop']] # Min/Max in y to search with slide_window()

    example_images = ['examples/sample.jpg']

    for img_src in example_images:

        img_boxes = []
        t1 = time.time()
        count = 0
        img = mpimg.imread(img_src)

        draw_img, heatmap, img_boxes, count = find_cars(img, config['scale'], config['y_start'], config['y_stop'],
                                                        config['orient'], config['pix_per_cell'],
                                                        config['cell_per_block'], config['window'],
                                                        config['image_height'], config['image_width'],
                                                        config['histogram_bins'], config['box_color_red'],
                                                        config['box_color_green'], config['box_color_blue'], X_scaler)

        labels = label(heatmap)
        logger.info("Heatmap max: " + str(np.max(heatmap)))
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        logger.info(str(time.time()-t) +' seconds to run, total windows = ' + str(count))
        out_maps = []
        out_boxes = []
        out_images.append(draw_img)

        out_titles.append(img_src.split('/')[-1])
        out_titles.append(img_src.split('/')[-1])

        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)

    fig = plt.figure(figsize=(12, 24))
    visualize(fig, 8, 2, out_images, out_titles, config['heatmap_vis_filename'])

    clip = VideoFileClip(config['input_video_file'])
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(config['output_video_file'], audio=False)

    sys.exit(0)
