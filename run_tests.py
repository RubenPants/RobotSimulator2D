"""
run_tests.py

Run all the tests.
"""
import unittest

if __name__ == '__main__':
    suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
