import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

# Add the project root directory to Python path
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DS.Visualisation.visg import VisG

class TestVisG(unittest.TestCase):
    _visg = None
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        if cls._visg is None:
            cls._visg = VisG()
    
    @property
    def visg(self):
        """Ensure visg is initialized."""
        if self._visg is None:
            self.__class__.setUpClass()
        return self._visg
        
    def test_simple_graph_visualization(self):
        """Test creating and visualizing a simple graph."""
        # Create a simple binary image (3x3)
        test_image = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)
        
        # Convert image to graph
        graph = self.visg.img2graph(test_image)
        
        # Test that the graph was created successfully
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, nx.DiGraph)
        
        # Test different layouts
        layouts = ['spring', 'circular', 'grid_square']
        for layout in layouts:
            plt.figure(figsize=(8, 6))
            self.visg.get_layout(layout=layout)
            self.visg.draw(title=f"Test Graph with {layout} layout")
            plt.show()
            
    def test_special_nodes(self):
        """Test adding special nodes to the graph."""
        # Create a simple binary image
        test_image = np.array([
            [0, 1],
            [1, 1]
        ], dtype=np.uint8)
        
        # Convert image to graph
        self.visg.img2graph(test_image)
        
        # Add special nodes
        self.visg.add_special_nodes()
        
        # Visualize with special nodes
        plt.figure(figsize=(8, 6))
        self.visg.get_layout(layout='spring')
        self.visg.draw(title="Graph with Special Nodes")
        plt.show()

def run_tests():
    """Run the tests directly."""
    test = TestVisG()
    TestVisG.setUpClass()
    test.test_simple_graph_visualization()
    test.test_special_nodes()

if __name__ == '__main__':
    run_tests()