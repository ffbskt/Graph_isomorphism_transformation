"""Test script for VisG package with different configurations."""

import numpy as np
import cv2
from visg import VisG, NODE_SHAPES, GRAPHVIZ_COLORS

def create_test_image1():
    """Create a simple cross pattern."""
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 1  # Horizontal line
    img[:, 2] = 1  # Vertical line
    return img

def create_test_image2():
    """Create a simple box pattern."""
    img = np.zeros((5, 5), dtype=np.uint8)
    img[1:4, 1:4] = 1  # Box
    img[2, 2] = 0  # Hole in middle
    return img

def test_image1():
    """Test first image with grid_square layout and custom red/blue configuration."""
    # Create custom configuration for cross pattern
    custom_config = {
        'singleton': {
            'shape': NODE_SHAPES['star'],
            'color': GRAPHVIZ_COLORS['red'],
            'size': 800
        },
        'coordinate': {
            'shape': NODE_SHAPES['pentagon'],
            'color': GRAPHVIZ_COLORS['lightblue'],
            'size': 600
        },
        'T': {
            'shape': NODE_SHAPES['triangle'],
            'color': GRAPHVIZ_COLORS['blue'],
            'size': 400
        },
        'F': {
            'shape': NODE_SHAPES['circle'],
            'color': GRAPHVIZ_COLORS['lightgray'],
            'size': 400
        },
        'reward_outside': {
            'shape': NODE_SHAPES['hexagon'],
            'color': GRAPHVIZ_COLORS['violet'],
            'size': 600
        },
        'reward_inside': {
            'shape': NODE_SHAPES['diamond'],
            'color': GRAPHVIZ_COLORS['cyan'],
            'size': 600
        },
        'default': {
            'shape': NODE_SHAPES['circle'],
            'color': GRAPHVIZ_COLORS['lightgray'],
            'size': 300
        }
    }

    # Create image and visualizer
    img1 = create_test_image1()
    vis = VisG(node_configs=custom_config)
    
    # Process and visualize
    G = vis.img2graph(img1)
    vis.add_special_nodes()
    vis.draw(layout='grid_square', title='Cross Pattern - Grid Square Layout')
    vis.save('test_image1_grid.png')
    vis.show()

def test_image2():
    """Test second image with hamiltonian layout and custom green/yellow configuration."""
    # Create custom configuration for box pattern
    custom_config = {
        'singleton': {
            'shape': NODE_SHAPES['octagon'],
            'color': GRAPHVIZ_COLORS['green'],
            'size': 800
        },
        'coordinate': {
            'shape': NODE_SHAPES['diamond'],
            'color': GRAPHVIZ_COLORS['yellow'],
            'size': 600
        },
        'T': {
            'shape': NODE_SHAPES['square'],
            'color': GRAPHVIZ_COLORS['forestgreen'],
            'size': 400
        },
        'F': {
            'shape': NODE_SHAPES['circle'],
            'color': GRAPHVIZ_COLORS['lightyellow'],
            'size': 400
        },
        # 'reward_outside': {
        #     'shape': NODE_SHAPES['plus'],
        #     'color': GRAPHVIZ_COLORS['tan'],
        #     'size': 600
        # },
        # 'reward_inside': {
        #     'shape': NODE_SHAPES['triangle_down'],
        #     'color': GRAPHVIZ_COLORS['maroon'],
        #     'size': 600
        # },
        # 'default': {
        #     'shape': NODE_SHAPES['circle'],
        #     'color': GRAPHVIZ_COLORS['gray'],
        #     'size': 300
        # }
    }

    # Create image and visualizer
    img2 = create_test_image2()
    vis = VisG(node_configs=custom_config, edge_configs={'sss':{}}, merge_configs=True)
    
    # Process and visualize
    G = vis.img2graph(img2)
    vis.add_special_nodes()
    vis.draw(layout='spring', title='Box Pattern - Hamiltonian Layout')
    vis.save('test_image2_hamiltonian.png')
    vis.show()

if __name__ == "__main__":
    print("Testing VisG with cross pattern...")
    test_image1()
    
    print("\nTesting VisG with box pattern...")
    test_image2() 