import unittest
import vehicle_detection as vehdetect


class TestVehicleDetection(unittest.TestCase):

    def setUp(self):
        self.config = vehdetect.load_config('test_configuration.json')

    def test_get_image_file_paths(self):
        path = self.config['nonvehicles_image_directory']
        image_list = vehdetect.get_image_file_paths(path)
        self.assertEqual(12, len(image_list))

    def tearDown(self):
        del self.config

if __name__ == '__main__':
    unittest.main()
