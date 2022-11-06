import unittest

from sklearn.utils.estimator_checks import check_estimator
from classifier.choquet_classifier import ChoquetClassifier


class TestChoquetClassifier(unittest.TestCase):
    def test_compatibility(self):
        cc = ChoquetClassifier()
        check_estimator(cc)

    def test_documentation(self):
        X = [[1, 3, 2], [1, 0, 3]]
        y = [1, 0]

        # test data
        Z = [[1, 1, 2], [2, 1, 3]]

        # first example
        cc = ChoquetClassifier()
        cc.fit(X, y)
        result = cc.predict(Z)

        self.assertListEqual([0, 0], list(result))

        # second example
        cc = ChoquetClassifier(additivity=3, regularization=1)
        cc.fit(X, y)
        result = cc.predict(Z)

        self.assertListEqual([0, 1], list(result))

        # third example
        y = [2, 1]
        cc = ChoquetClassifier()
        cc.fit(X, y)
        result = cc.predict(Z)

        self.assertListEqual([1, 1], list(result))

        y = ['two', 'one']
        cc = ChoquetClassifier()
        cc.fit(X, y)
        result = cc.predict(Z)

        self.assertListEqual(['one', 'one'], list(result))


if __name__ == '__main__':
    unittest.main()
