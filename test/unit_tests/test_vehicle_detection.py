import unittest
import vehicle_detection as vehdetect
import matplotlib.image as mpimg
import cv2


class TestVehicleDetection(unittest.TestCase):

    def setUp(self):
        self.config = vehdetect.load_config('test_configuration.json')
        self.image_path = '../../test_images/vehicles/GTI_Left/image0009.png'

    def test_get_image_file_paths(self):
        path = self.config['nonvehicles_image_directory']
        image_list = vehdetect.get_image_file_paths(path)
        self.assertEqual(12, len(image_list))

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


    def tearDown(self):
        del self.config


if __name__ == '__main__':
    unittest.main()
