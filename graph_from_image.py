"""Module for converting images to graph representations."""

import networkx as nx
import numpy as np
import cv2

def img2graph_binary(img):
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
        G.add_node(f'{iv}i', type='coordinate', label=f'R{iv}')  # Row nodes as squares
        
        for jv in range(width):
            # Add column node if not exists
            if not G.has_node(f'{jv}j'):
                G.add_node(f'{jv}j', type='coordinate', label=f'C{jv}')  # Column nodes as squares
            
            # Add pixel node
            pixel_node = f'p_{iv}_{jv}'
            pixel_value = int(img[iv, jv])
            # Pixel nodes as triangles, with value in label
            if pixel_value == 1:
                G.add_node(pixel_node, type='T', label=f'{pixel_value}')
            else:
                G.add_node(pixel_node, type='F', label=f'{pixel_value}')
            
            # Connect pixel to row and column
            G.add_edge(pixel_node, f'{iv}i', type='black')
            G.add_edge(f'{iv}i', pixel_node, type='black')
            G.add_edge(pixel_node, f'{jv}j', type='black')
            G.add_edge(f'{jv}j', pixel_node, type='black')
            
            # Connect to neighboring pixels
            if iv > 0:  # Connect to pixel above
                G.add_edge(f'p_{iv-1}_{jv}', pixel_node, type='black')
                G.add_edge(pixel_node, f'p_{iv-1}_{jv}', type='black')
            
            if jv > 0:  # Connect to pixel on the left
                G.add_edge(f'p_{iv}_{jv-1}', pixel_node, type='black')
                G.add_edge(pixel_node, f'p_{iv}_{jv-1}', type='black')
            
            if iv > 0 and jv > 0:  # Connect to diagonal pixel
                G.add_edge(f'p_{iv-1}_{jv-1}', pixel_node, type='black')
    
    return G

def add_special_nodes(G, special_nodes, special_edges):
    """
    Add special nodes and their edges to the graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph to add nodes to
    special_nodes : list of tuples
        List of (node_id, node_attributes) tuples to add
    special_edges : list of tuples
        List of (source, target, edge_attributes) tuples to add
        
    Returns:
    --------
    networkx.DiGraph
        Graph with added special nodes and edges
    """
    # Add special nodes
    G.add_nodes_from(special_nodes)
    
    # Add special edges
    G.add_edges_from(special_edges)
    
    return G

def get_default_special_nodes(G):
    """
    Get default special nodes and edges configuration.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Input graph to configure special nodes for
        
    Returns:
    --------
    tuple
        (special_nodes, special_edges) where:
        - special_nodes is a list of (node_id, node_attributes) tuples
        - special_edges is a list of (source, target, edge_attributes) tuples
    """
    special_nodes = [
        ('s', {'type': 'singleton', 'label': 'Start'}),
        ('b', {'type': 'singleton', 'label': 'Boundary'}),
        ('c_0', {'type': 'reward_outside', 'label': 'cls0'}),
        ('c_1', {'type': 'reward_inside', 'label': 'cls1'}),
    ]
    
    special_edges = []
    
    # Connect boundary to pixels with value 1
    for node in G.nodes():
        if node.startswith('p_') and G.nodes[node]['label'] == '1':
            special_edges.extend([
                (node, 'b', {'type': 'white'}),
                ('b', node, {'type': 'white'})
            ])
    
    # Add start node connection
    special_edges.extend([
        ('s', 'p_1_0', {'type': 'white', 'label': None}),
        ('p_1_0', 's', {'type': 'white', 'label': None}),
        ('s', 'c_0', {'type': 'black', 'label': None}),
        ('s', 'c_1', {'type': 'black', 'label': None})
    ])
    
    return special_nodes, special_edges