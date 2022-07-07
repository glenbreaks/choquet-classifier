import unittest

from sklearn.utils.estimator_checks import check_estimator
from classifier.choquet_classifier import ChoquetClassifier

class TestChoquetClassifier(unittest.TestCase):
    def test_compatibility(self):
        check_estimator(ChoquetClassifier())


if __name__ == '__main__':
    unittest.main()
