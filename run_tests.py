"""
run_tests.py

Run all the tests.
"""
import sys
import unittest

if __name__ == '__main__':
    if sys.platform == 'linux':
        suite_py = unittest.TestLoader().discover('.', pattern="*_test.py")
        suite_cy = unittest.TestLoader().discover('.', pattern="*_test_cy.py")
        suite = unittest.TestSuite([suite_py, suite_cy])
    else:
        suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
