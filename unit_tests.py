__author__ = 'mingles'

import unittest
import test


class MyTestCase(unittest.TestCase):
    def knn_mode(self):
        self.assertEquals(test.knn_mode([1,2,3]), 1)

if __name__ == '__main__':
    unittest.main()
