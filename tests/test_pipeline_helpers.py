"""Small tests for feature extraction helpers; these use artificial arrays only."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.extract_features import extract_one


class FeatureExtractionTests(unittest.TestCase):
    def test_range_feature(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.txt"
            np.savetxt(path, np.array([[1.0, 5.0], [3.0, 2.0], [4.0, 9.0]]))
            features = extract_one(path, time_length=None)
            self.assertAlmostEqual(features["region_1_range"], 3.0)
            self.assertAlmostEqual(features["region_2_range"], 7.0)

    def test_rejects_non_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.txt"
            np.savetxt(path, np.array([1.0, 2.0, 3.0]))
            with self.assertRaises(ValueError):
                extract_one(path, time_length=None)


if __name__ == "__main__":
    unittest.main()
