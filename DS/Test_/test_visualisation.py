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
        
        # Test different layouts and save to a combined image
        layouts = ['spring', 'circular', 'grid_square']
        fig, axs = plt.subplots(1, len(layouts), figsize=(15, 5))
        
        for i, layout in enumerate(layouts):
            self.visg.get_layout(layout=layout)
            self.visg.draw(title=f"Test Graph with {layout} layout", ax=axs[i])
        
        # Save the combined visualization
        output_dir = "test_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "visualization_layouts.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)
            
    def test_special_nodes(self):
        """Test adding special nodes to the graph."""
        # Create a simple binary image
        test_image = np.array([
            [0, 1],
            [1, 1]
        ], dtype=np.uint8)
        
        # Convert image to graph
        self.visg.img2graph(test_image)
        
        # Create a figure with two subplots - before and after adding special nodes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Draw the graph before adding special nodes
        self.visg.get_layout(layout='spring')
        self.visg.draw(title="Graph without Special Nodes", ax=ax1)
        
        # Add special nodes
        self.visg.add_special_nodes()
        
        # Draw the graph with special nodes
        self.visg.get_layout(layout='spring')
        self.visg.draw(title="Graph with Special Nodes", ax=ax2)
        
        # Save the visualization
        output_dir = "test_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "visualization_special_nodes.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Generate a combined visualization of all test results
        output_dir = "test_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # List all visualization files
        visualization_files = [
            os.path.join(output_dir, "visualization_layouts.png"),
            os.path.join(output_dir, "visualization_special_nodes.png")
        ]
        
        # Create a combined figure
        combined_fig = plt.figure(figsize=(15, 10))
        
        # Add each visualization as a subplot
        for i, file_path in enumerate(visualization_files):
            if os.path.exists(file_path):
                img = plt.imread(file_path)
                ax = combined_fig.add_subplot(len(visualization_files), 1, i + 1)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(os.path.basename(file_path).replace('.png', '').replace('_', ' ').title())
        
        # Save the combined figure
        combined_fig.tight_layout()
        combined_fig.savefig(os.path.join(output_dir, "combined_visualizations.png"), bbox_inches='tight', dpi=300)
        plt.close(combined_fig)
        
        print(f"Saved combined visualization to {os.path.join(output_dir, 'combined_visualizations.png')}")

def run_tests():
    """Run the tests directly."""
    test = TestVisG()
    TestVisG.setUpClass()
    test.test_simple_graph_visualization()
    test.test_special_nodes()
    TestVisG.tearDownClass()

if __name__ == '__main__':
    run_tests()