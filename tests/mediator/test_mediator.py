import unittest
from classifier.mediator.mediator import Mediator
from sklearn.utils.validation import check_array

class TestMediator(unittest.TestCase):
    def test_fit_components(self):
        X = [[1, 3, 2], [2, 1, 3]]
        y = [0, 1]

        Z = [[3, 2, 1],[1, 2, 3]]
        mediator = Mediator()
        X = check_array(X)
        Z = check_array(Z)
        mediator.fit_components(X, y, 3, 1)
        predict = mediator.predict_classes(Z)
        print(predict, mediator.parameters)

if __name__ == '__main__':
    unittest.main()