import unittest

from sklearn.utils.estimator_checks import check_estimator
from classifier.choquet_classifier import ChoquetClassifier

# TODO: make check_estimator attainable
class TestChoquetClassifier(unittest.TestCase):
    def test_compatibility(self):
        cc = ChoquetClassifier()
        check_estimator(cc)

    def test_examples(self):
        X = [[1, 3, 2], [2, 1, 3]]
        y = [0, 1]
        Z = [[3, 2, 1], [1, 2, 3]]
        cc = ChoquetClassifier(additivity=3)
        cc.fit(X, y)
        result = cc.predict(Z)
        print(result)


if __name__ == '__main__':
    unittest.main()
