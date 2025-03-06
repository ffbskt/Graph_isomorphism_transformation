import unittest
import numpy as np
import networkx as nx
from Visualisation.visg import VisG
import matplotlib.pyplot as plt

class TestVisG(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.visg = VisG()
        
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
            plt.savefig(f"Test/test_graph_{layout}.png")
            plt.close()
            
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
        plt.savefig("Test/test_graph_special_nodes.png")
        plt.close()

if __name__ == '__main__':
    unittest.main() 