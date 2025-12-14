"""Unit tests for utility functions."""

import unittest

from firenet.utils.normalization import normalize, inverse_normalize


class TestNormalization(unittest.TestCase):
    """Test cases for normalization functions."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        self.assertEqual(normalize(50, 0, 100), 0.5)
        self.assertEqual(normalize(0, 0, 100), 0.0)
        self.assertEqual(normalize(100, 0, 100), 1.0)
    
    def test_normalize_clamping(self):
        """Test that values are clamped to range."""
        self.assertEqual(normalize(150, 0, 100), 1.0)
        self.assertEqual(normalize(-50, 0, 100), 0.0)
    
    def test_normalize_negative_range(self):
        """Test normalization with negative values."""
        self.assertEqual(normalize(0, -10, 10), 0.5)
        self.assertEqual(normalize(-10, -10, 10), 0.0)
        self.assertEqual(normalize(10, -10, 10), 1.0)
    
    def test_normalize_invalid_range(self):
        """Test that equal min/max raises error."""
        with self.assertRaises(ValueError):
            normalize(5, 5, 5)
    
    def test_inverse_normalize_basic(self):
        """Test basic inverse normalization."""
        self.assertEqual(inverse_normalize(50, 0, 100), 0.5)
        self.assertEqual(inverse_normalize(0, 0, 100), 1.0)
        self.assertEqual(inverse_normalize(100, 0, 100), 0.0)
    
    def test_inverse_normalize_clamping(self):
        """Test that inverse values are clamped."""
        self.assertEqual(inverse_normalize(150, 0, 100), 0.0)
        self.assertEqual(inverse_normalize(-50, 0, 100), 1.0)
    
    def test_normalize_float_precision(self):
        """Test float precision in normalization."""
        result = normalize(33.33, 0, 100)
        self.assertAlmostEqual(result, 0.3333, places=4)


if __name__ == "__main__":
    unittest.main()
