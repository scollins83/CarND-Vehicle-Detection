import unittest
import vehicle_detection as vehdetect
import matplotlib.image as mpimg
import cv2


class TestVehicleDetection(unittest.TestCase):

    def setUp(self):
        self.config = vehdetect.load_config('test_configuration.json')
        self.image_path = '../../test_images/vehicles/GTI_Left/image0009.png'
        self.image_path_list = ['../../test_images/vehicles/GTI_Left/image0009.png',
                                '../../test_images/vehicles/GTI_Left/image0010.png',
                                '../../test_images/vehicles/GTI_Left/image0011.png']

    def test_get_image_file_paths(self):
        path = self.config['nonvehicles_image_directory']
        image_list = vehdetect.get_image_file_paths(path)
        self.assertEqual(len(image_list), 12)

    def test_get_hog_features(self):
        image = mpimg.imread(self.image_path)
        hog_image = image[:, :, self.config['hog_channel']]
        test_hog_features = vehdetect.get_hog_features(hog_image,
                                                       self.config['orient'],
                                                       self.config['pix_per_cell'],
                                                       self.config['cell_per_block'],
                                                       self.config['visualize_hog'],
                                                       self.config['feature_vec_hog'])
        self.assertEqual(test_hog_features.shape[0], 1764)

    def test_bin_spatial(self):
        image = mpimg.imread(self.image_path)
        bin_spatial = vehdetect.bin_spatial(image,
                                            (self.config['image_height'],
                                             self.config['image_width']))
        self.assertEqual(len(bin_spatial), 3072)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.assertRaises(AssertionError,
                          vehdetect.bin_spatial,
                          gray_image,
                          (self.config['image_height'],
                           self.config['image_width']))

    def test_color_histogram(self):
        image = mpimg.imread(self.image_path)
        histogram_features = vehdetect.color_histogram(image,
                                                       self.config['histogram_bins'])
        self.assertEqual(len(histogram_features), 96)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.assertRaises(AssertionError,
                          vehdetect.color_histogram,
                          gray_image,
                          self.config['histogram_bins'])

    def test_extract_features(self):
        feature_list = vehdetect.extract_features(self.image_path_list,
                                                  self.config['color_space'],
                                                  (self.config['image_height'],
                                                   self.config['image_width']),
                                                  self.config['histogram_bins'],
                                                  self.config['orient'],
                                                  self.config['pix_per_cell'],
                                                  self.config['cell_per_block'],
                                                  self.config['hog_channel'],
                                                  self.config['extract_spatial_features'],
                                                  self.config['extract_histogram_features'],
                                                  self.config['extract_hog_features'],
                                                  self.config['visualize_hog'],
                                                  self.config['feature_vec_hog'])
        self.assertEqual(len(feature_list), 3)
        self.assertTupleEqual(feature_list[0].shape, (4932,))
        self.assertTupleEqual(feature_list[1].shape, (4932,))
        self.assertTupleEqual(feature_list[2].shape, (4932,))

    def test_slide_window(self):
        image = mpimg.imread(self.image_path)
        window_list = vehdetect.slide_window(image,
                                             [None, None],
                                             [None, None],
                                             (self.config['xy_window_x'], self.config['xy_window_y']),
                                             (self.config['xy_overlap_x'], self.config['xy_overlap_y']))
        windows = [((0, 0), (32, 32)),
                   ((16, 0), (48, 32)),
                   ((32, 0), (64, 32)),
                   ((0, 16), (32, 48)),
                   ((16, 16), (48, 48)),
                   ((32, 16), (64, 48)),
                   ((0, 32), (32, 64)),
                   ((16, 32), (48, 64)),
                   ((32, 32), (64, 64))]
        self.assertListEqual(window_list, windows)

    def test_draw_boxes(self):
        image = mpimg.imread(self.image_path)
        windows = [((0, 0), (32, 32)),
                   ((16, 0), (48, 32)),
                   ((32, 0), (64, 32)),
                   ((0, 16), (32, 48)),
                   ((16, 16), (48, 48)),
                   ((32, 16), (64, 48)),
                   ((0, 32), (32, 64)),
                   ((16, 32), (48, 64)),
                   ((32, 32), (64, 64))]
        color = (255, 0, 0)
        thickness = 6
        boxed_image = vehdetect.draw_boxes(image, windows, color, thickness)
        self.assertNotEqual(boxed_image.all(), image.all())

    def test_extract_single_image_features(self):
        image = mpimg.imread(self.image_path)
        features = vehdetect.extract_single_image_features(image,
                                                           self.config['color_space'],
                                                           (self.config['image_height'],
                                                            self.config['image_width']),
                                                           self.config['histogram_bins'],
                                                           self.config['orient'],
                                                           self.config['pix_per_cell'],
                                                           self.config['cell_per_block'],
                                                           self.config['hog_channel'],
                                                           self.config['extract_spatial_features'],
                                                           self.config['extract_histogram_features'],
                                                           self.config['extract_hog_features'],
                                                           self.config['visualize_hog'],
                                                           self.config['feature_vec_hog'])
        self.assertTupleEqual(features.shape, (4932,))

    @unittest.skip
    def test_search_windows(self):
        image = mpimg.imread(self.image_path)
        windows = [((0, 0), (32, 32)),
                   ((16, 0), (48, 32)),
                   ((32, 0), (64, 32)),
                   ((0, 16), (32, 48)),
                   ((16, 16), (48, 48)),
                   ((32, 16), (64, 48)),
                   ((0, 32), (32, 64)),
                   ((16, 32), (48, 64)),
                   ((32, 32), (64, 64))]
        classifier = None
        scaler = None
        positive_windows = vehdetect.search_windows(image,
                                                    windows,
                                                    self.config['color_space'],
                                                    (self.config['image_height'],
                                                     self.config['image_width']),
                                                    self.config['histogram_bins'],
                                                    self.config['orient'],
                                                    self.config['pix_per_cell'],
                                                    self.config['cell_per_block'],
                                                    self.config['hog_channel'],
                                                    self.config['extract_spatial_features'],
                                                    self.config['extract_histogram_features'],
                                                    self.config['extract_hog_features'])



    def tearDown(self):
        del self.config


if __name__ == '__main__':
    unittest.main()
