"""Main VisG class for graph visualization and manipulation."""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2

from .constants import (
    NODE_SHAPES, GRAPHVIZ_COLORS, 
    DEFAULT_NODE_CONFIGS, DEFAULT_EDGE_CONFIGS,
    LAYOUT_CONFIGS
)
from .layouts import grid_square_layout, HamiltonianLayout

class VisG:
    def __init__(self, node_configs=None, edge_configs=None, merge_configs=True):
        """
        Initialize the VisG with custom or default configurations.
        
        Parameters:
        -----------
        node_configs : dict, optional
            Custom node configurations to override defaults
        edge_configs : dict, optional
            Custom edge configurations to override defaults
        """
        self.node_configs = node_configs or DEFAULT_NODE_CONFIGS
        self.edge_configs = edge_configs or DEFAULT_EDGE_CONFIGS
        if merge_configs:
            self.merge_configs(node_configs, edge_configs)

        self.pos = None
        self.graph = None

    def merge_configs(self, node_configs=None, edge_configs=None):
        """
        Merge custom node and edge configurations with defaults.
        
        Parameters:
        
        """
        if node_configs is not None:
            for key, value in DEFAULT_NODE_CONFIGS.items():
                if key not in node_configs:
                    self.node_configs[key] = value
        if edge_configs is not None:
            for key, value in DEFAULT_EDGE_CONFIGS.items():
                if key not in edge_configs:
                    self.edge_configs[key] = value


    def img2graph(self, img):
        """
        Convert an image to a graph representation with row, column and pixel nodes.
        Creates nodes of coordinate (row/col) and T/F (pixels).
        
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
            G.add_node(f'{iv}i', type='coordinate', label=f'R{iv}')
            
            for jv in range(width):
                # Add column node if not exists
                if not G.has_node(f'{jv}j'):
                    G.add_node(f'{jv}j', type='coordinate', label=f'C{jv}')
                
                # Add pixel node
                pixel_node = f'p_{iv}_{jv}'
                pixel_value = int(img[iv, jv])
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
        
        self.graph = G
        return G
    
    def add_graph(self, graph):
        """
        Add an existing graph to the VisG instance.
        
        Parameters:
        -----------
        graph : networkx.DiGraph
            Graph to add
        """
        self.graph = graph
        return graph
    

    def add_special_nodes(self, special_nodes=None, special_edges=None):
        """
        Add special nodes and their edges to the graph.
        
        Parameters:
        -----------
        special_nodes : list of tuples
            List of (node_id, node_attributes) tuples to add
        special_edges : list of tuples
            List of (source, target, edge_attributes) tuples to add
        """
        if self.graph is None:
            raise ValueError("No graph exists. Create a graph first using img2graph or set_graph.")

        if special_nodes is None and special_edges is None:
            special_nodes, special_edges = self.get_default_special_nodes()

        # Add special nodes
        self.graph.add_nodes_from(special_nodes)
        
        # Add special edges
        self.graph.add_edges_from(special_edges)
        
        return self.graph

    def get_default_special_nodes(self):
        """Get default special nodes and edges configuration."""
        if self.graph is None:
            raise ValueError("No graph exists. Create a graph first using img2graph or set_graph.")

        special_nodes = [
            ('s', {'type': 'singleton', 'label': 'Start'}),
            ('b', {'type': 'singleton', 'label': 'Boundary'}),
            ('c_0', {'type': 'reward_outside', 'label': 'cls0'}),
            ('c_1', {'type': 'reward_inside', 'label': 'cls1'}),
        ]
        
        special_edges = []
        
        # Connect boundary to pixels with value 1
        for node in self.graph.nodes():
            if node.startswith('p_') and self.graph.nodes[node]['label'] == '1':
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

    def set_graph(self, graph):
        """Set an existing graph for visualization."""
        self.graph = graph

    def get_layout(self, layout='spring', pos=None):
        """
        Get the layout for the graph.
        
        Parameters:
        -----------
        layout : str
            Layout algorithm to use if pos is None
            Options: 'spring', 'circular', 'random', 'shell', 'spectral', 
                    'kamada_kawai', 'planar', 'grid_square', 'hamiltonian'
        pos : dict, optional
            Pre-computed positions of nodes
        """
        if pos is not None:
            self.pos = pos
            return pos
            
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'planar': nx.planar_layout,
            'grid_square': grid_square_layout,
            'hamiltonian': lambda G: HamiltonianLayout(G).get_pos()
        }
        
        layout_func = layout_funcs.get(layout, nx.spring_layout)
        
        if layout == 'hamiltonian':
            self.pos = layout_func(self.graph)
        else:
            layout_params = LAYOUT_CONFIGS.get(layout, {})
            self.pos = layout_func(self.graph, **layout_params)
            
        return self.pos

    def draw(self, layout='spring', title="Graph Visualization", figsize=(10, 8), ax=None):
        """
        Draw the graph with custom node and edge attributes.
        
        Parameters:
        -----------
        layout : str
            Layout algorithm to use if pos is None
        title : str
            Title of the visualization
        figsize : tuple
            Size of the figure (width, height)
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, creates new figure
        """
        if self.graph is None:
            raise ValueError("No graph exists. Create a graph first using img2graph or set_graph.")

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        
        # Get node positions if not already computed
        if self.pos is None:
            self.get_layout(layout)
        
        # Set axes limits with margin
        margin = 0.2
        all_coords = np.array(list(self.pos.values()))
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        ax.set_xlim(x_min - margin * width, x_max + margin * width)
        ax.set_ylim(y_min - margin * height, y_max + margin * height)
        
        # Draw nodes for each type
        for node_type in set(nx.get_node_attributes(self.graph, 'type').values()):
            # Get nodes of current type
            node_list = [node for node, attr in self.graph.nodes(data=True) 
                        if attr.get('type', 'default') == node_type]
            
            if not node_list:
                continue
                
            config = self.node_configs.get(node_type, self.node_configs['default'])
            nx.draw_networkx_nodes(
                self.graph, self.pos,
                nodelist=node_list,
                node_color=[config['color']],
                node_shape=config['shape'],
                node_size=config['size'],
                ax=ax
            )
        
        # Draw edges for each type
        for edge_type in set(nx.get_edge_attributes(self.graph, 'type').values()):
            edge_list = [(u, v) for u, v, attr in self.graph.edges(data=True) 
                        if attr.get('type', 'default') == edge_type]
            
            if not edge_list:
                continue
                
            config = self.edge_configs.get(edge_type, self.edge_configs['default'])
            nx.draw_networkx_edges(
                self.graph, self.pos,
                edgelist=edge_list,
                edge_color=config['color'],
                width=config['width'],
                ax=ax
            )
        
        # Add node labels (small size)
        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx_labels(self.graph, self.pos, labels, font_size=8, ax=ax)
        
        # Add edge labels (small size)
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels, font_size=6)
        
        # Add legend for node types
        legend_elements = [
            mpatches.Patch(
                color=self.node_configs[node_type]['color'],
                label=f'Node {node_type}'
            )
            for node_type in self.node_configs
            if node_type != 'default'
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_title(title)
        ax.axis('off')
        
    def show(self):
        """Display the graph."""
        plt.show()
        
    def save(self, filename):
        """Save the graph visualization to a file."""
        plt.savefig(filename, bbox_inches='tight', dpi=300) 