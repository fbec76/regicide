#!/usr/bin/env python3
import unittest
import sys

def run_tests():
    """Run all tests in the tests directory."""
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover('.')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())