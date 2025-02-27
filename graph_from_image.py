"""Module for converting images to graph representations."""

import networkx as nx
import numpy as np
import cv2

def img2graph(img):
    """
    Convert an image to a graph representation with row, column and pixel nodes.
    Creates nodes of type2 (row/col) and type3 (pixels).
    
    Parameters:
    -----------
    img : numpy.ndarray or str
        Input image (binary) or path to image file
        
    Returns:
    --------
    networkx.DiGraph
        Graph with row, column and pixel nodes
    """
    # Preprocess image if needed
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary values
    img = (img > 0).astype(np.uint8)
    
    if isinstance(img, list):
        img = img[0]  # Take first image if list provided
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Get image dimensions
    height, width = img.shape
    
    # Add nodes and edges for each pixel
    for iv in range(height):
        # Add row node
        G.add_node(f'{iv}i', type='type2', label=f'R{iv}')  # Row nodes as squares
        
        for jv in range(width):
            # Add column node if not exists
            if not G.has_node(f'{jv}j'):
                G.add_node(f'{jv}j', type='type2', label=f'C{jv}')  # Column nodes as squares
            
            # Add pixel node
            pixel_node = f'p_{iv}_{jv}'
            pixel_value = int(img[iv, jv])
            # Pixel nodes as triangles, with value in label
            G.add_node(pixel_node, type='type3', label=f'{pixel_value}')
            
            # Connect pixel to row and column
            G.add_edge(pixel_node, f'{iv}i', type='black')
            G.add_edge(f'{iv}i', pixel_node, type='black')
            G.add_edge(pixel_node, f'{jv}j', type='black')
            G.add_edge(f'{jv}j', pixel_node, type='black')
            
            # Connect to neighboring pixels
            if iv > 0:  # Connect to pixel above
                G.add_edge(f'p_{iv-1}_{jv}', pixel_node, type='white')
                G.add_edge(pixel_node, f'p_{iv-1}_{jv}', type='white')
            
            if jv > 0:  # Connect to pixel on the left
                G.add_edge(f'p_{iv}_{jv-1}', pixel_node, type='white')
                G.add_edge(pixel_node, f'p_{iv}_{jv-1}', type='white')
            
            if iv > 0 and jv > 0:  # Connect to diagonal pixel
                G.add_edge(f'p_{iv-1}_{jv-1}', pixel_node, type='white')
    
    return G

def add_special_nodes(G, n_classes=1, start_node='s'):
    """
    Add special nodes (type1) and their edges to the graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph to add nodes to
    n_classes : int
        Number of class nodes to add
    start_node : str
        Name of the start node
        
    Returns:
    --------
    networkx.DiGraph
        Graph with added special nodes and edges
    """
    # Add special nodes
    G.add_node('s', type='type1', label='Start')
    G.add_node('b', type='type1', label='Boundary')
    
    # Connect boundary to pixels with value 1
    for node in G.nodes():
        if node.startswith('p_') and G.nodes[node]['label'] == '1':
            G.add_edge(node, 'b', type='black')
            G.add_edge('b', node, type='black')
    
    # Add start node connection
    G.add_edge('s', 'p_1_0', type='black')
    G.add_edge('p_1_0', 's', type='black')
    
    # Add class nodes
    for i in range(n_classes):
        class_node = f'c_{i}'
        G.add_node(class_node, type='type2', label=f'Class{i}')
        G.add_edge(start_node, class_node, type='black')
    
    return G