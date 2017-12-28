import unittest
import vehicle_detection as vehdetect


class TestVehicleDetection(unittest.TestCase):

    def setUp(self):
        self.config = vehdetect.load_config('test_configuration.json')

    def test_something(self):
        path = self.config['image_file_pattern']
        image_list = vehdetect.get_image_file_paths(path)
        self.assertEqual(len(image_list), 6)

    def tearDown(self):
        del self.config

if __name__ == '__main__':
    unittest.main()
