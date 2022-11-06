import unittest
from classifier.mediator.mediator import Mediator
from sklearn.utils.validation import check_array

class TestMediator(unittest.TestCase):

    def test_fit_components(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]

        additivity = 3

        Z = [[0.6, 0.5, 0.4],[0.1, 0.2, 0.3]]
        mediator = Mediator()
        X = check_array(X)
        Z = check_array(Z)
        mediator.fit_components(X, y, additivity, 1)
        predict = mediator.predict_classes(Z)
        print(predict, mediator.parameters)

if __name__ == '__main__':
    unittest.main()
