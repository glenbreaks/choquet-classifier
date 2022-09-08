from classifier.mediator.components.moebius_transform import MoebiusTransformation
import unittest


class TestMoebiusTransformation(unittest.TestCase):

    def test_powerset(self):
        s = {2, 4, 6, 8}
        x_data = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

        m = MoebiusTransformation(2, x_data, [])

        powerset = m.get_powerset(s)
        expected_powerset = [frozenset(), frozenset([2]), frozenset([4]), frozenset([6]), frozenset([8]),
                             frozenset([2, 4]),  frozenset([2, 6]),  frozenset([2, 8]),  frozenset([4, 6]),
                             frozenset([4, 8]),  frozenset([6, 8]), frozenset([2, 4, 6]),  frozenset([2, 4, 8]),
                             frozenset([2, 6, 8]),  frozenset([4, 6, 8]),  frozenset([2, 4, 6, 8])]

        self.assertCountEqual(list(powerset), expected_powerset)


if __name__ == '__main__':
    unittest.main()
