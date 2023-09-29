import unittest
import logging
from apriori import generate_subsets

class TestGenerateSubsets(unittest.TestCase):
    def test_generate_subsets(self):
        itemset = frozenset({'A', 'B', 'C'})
        expected_subsets = [
            frozenset(),
            frozenset({'A'}),
            frozenset({'B'}),
            frozenset({'C'}),
            frozenset({'A', 'B'}),
            frozenset({'A', 'C'}),
            frozenset({'B', 'C'}),
            frozenset({'A', 'B', 'C'})
        ]

        # Sort the subsets for comparison
        expected_subsets = sorted(expected_subsets, key=lambda x: (len(x), sorted(list(x))))
        result_subsets = sorted(generate_subsets(itemset), key=lambda x: (len(x), sorted(list(x))))

        self.assertEqual(result_subsets, expected_subsets)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run the tests
    unittest.main()
