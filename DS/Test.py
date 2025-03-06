import sys
import os
import Test_

from Test_.test_visualisation import TestVisG

def main():
    # Create test instance and run setup
    test = TestVisG()
    TestVisG.setUpClass()  # Call the class method directly
    
    # Run tests
    test.test_simple_graph_visualization()
    test.test_special_nodes()

if __name__ == "__main__":
    main()
