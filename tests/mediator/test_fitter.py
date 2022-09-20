import unittest
import numpy as np

from classifier.mediator.fitter import Fitter

class TestEstimator(unittest.TestCase):
	def test_feature_transformation(self):
		X = [[1, 7, 2, 2, 5],
			 [6, 2, 65, 2, 4],
			 [-7, 0.1, 55.3, 5, 10]]

		print(Fitter().fit_feature_transformation(X))

if __name__ == '__main__':
	unittest.main()