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
from Test_.test_transformation import TestTransformation
from Test_.test_visualisation import TestVisG

def main():
    print("\n===== Running DS Dynamic System Test Suite =====\n")
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    print("Loading test modules:")
    print("  - Matching Tests")
    suite.addTest(unittest.makeSuite(TestMatching))
    print("  - Transformation Tests")
    suite.addTest(unittest.makeSuite(TestTransformation))
    print("  - Visualization Tests")
    suite.addTest(unittest.makeSuite(TestVisG))
    
    print("\nRunning tests...\n")
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    print("Initializing DS test suite...")
    success = main()
    if success:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed!")
    # Exit with appropriate code
    sys.exit(0 if success else 1)
