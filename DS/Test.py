# import Test_
# from Test_.test_visualisation import TestVisG

# def main():
#     # Create test instance and run setup
#     test = TestVisG()
#     TestVisG.setUpClass()  # Call the class method directly
    
#     # Run tests
#     test.test_simple_graph_visualization()
#     test.test_special_nodes()

import unittest
import sys
import os

# Ensure Test_ directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the test cases
from Test_.test_matching import TestMatching

def main():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    suite.addTest(unittest.makeSuite(TestMatching))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = main()
    # Exit with appropriate code
    sys.exit(0 if success else 1)
