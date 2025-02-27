import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from graph_constants import (
    NODE_SHAPES, GRAPHVIZ_COLORS, 
    DEFAULT_NODE_CONFIGS, DEFAULT_EDGE_CONFIGS,
    LAYOUT_CONFIGS
)
from layout_funcs import grid_square_layout, HamiltonianLayout

class GraphVisualizer:
    def __init__(self, node_configs=None, edge_configs=None):
        """
        Initialize the visualizer with custom or default configurations.
        
        Parameters:
        -----------
        node_configs : dict, optional
            Custom node configurations to override defaults
        edge_configs : dict, optional
            Custom edge configurations to override defaults
        """
        self.node_configs = node_configs or DEFAULT_NODE_CONFIGS
        self.edge_configs = edge_configs or DEFAULT_EDGE_CONFIGS
        self.pos = None

    def get_pos(self):
        for k in self.pos:
            self.pos[k] = np.array(self.pos[k])
        return self.pos

    def get_layout(self, graph, pos=None, layout='spring'):
        """
        Get the layout for the graph.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The graph to get layout for
        pos : dict, optional
            Pre-computed positions of nodes
        layout : str, optional
            Layout algorithm to use if pos is None
            Options: 'spring', 'circular', 'random', 'shell', 'spectral', 'kamada_kawai', 'planar', 'grid_square', 'hamiltonian'
        """
        if pos is not None:
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
            return layout_func(graph)
        else:
            layout_params = LAYOUT_CONFIGS.get(layout, {})
            return layout_func(graph, **layout_params)

    def draw_graph(self, graph, pos=None, layout='spring', title="Graph Visualization", figsize=(10, 8), ax=None):
        """
        Draw the graph with custom node and edge attributes.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The graph to visualize. Nodes should have 'label' and 'type' attributes.
            Edges should have 'label' and 'type' attributes.
        pos : dict, optional
            Pre-computed positions of nodes. If None, uses specified layout.
        layout : str, optional
            Layout algorithm to use if pos is None
        title : str
            Title of the visualization
        figsize : tuple
            Size of the figure (width, height)
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, creates new figure
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        
        # Get node positions
        pos = self.get_layout(graph, pos, layout)
        self.pos = pos
        
        # Set axes limits with margin
        margin = 0.2
        all_coords = np.array(list(pos.values()))
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        ax.set_xlim(x_min - margin * width, x_max + margin * width)
        ax.set_ylim(y_min - margin * height, y_max + margin * height)
        
        # Draw nodes for each type
        for node_type in set(nx.get_node_attributes(graph, 'type').values()):
            # Get nodes of current type
            node_list = [node for node, attr in graph.nodes(data=True) 
                        if attr.get('type', 'default') == node_type]
            
            if not node_list:
                continue
                
            config = self.node_configs.get(node_type, self.node_configs['default'])
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=node_list,
                node_color=[config['color']],
                node_shape=config['shape'],
                node_size=config['size'],
                ax=ax
            )
        
        # Draw edges for each type
        for edge_type in set(nx.get_edge_attributes(graph, 'type').values()):
            edge_list = [(u, v) for u, v, attr in graph.edges(data=True) 
                        if attr.get('type', 'default') == edge_type]
            
            if not edge_list:
                continue
                
            config = self.edge_configs.get(edge_type, self.edge_configs['default'])
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=edge_list,
                edge_color=config['color'],
                width=config['width'],
                ax=ax
            )
        
        # Add node labels (small size)
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        # Add edge labels (small size)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)
        
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

def create_molecule_graph():
    """Create an example graph representing a simple molecule structure."""
    G = nx.Graph()
    
    # Add atoms as nodes
    atoms = [
        ('C1', {'type': 'type2', 'label': 'C'}),  # Carbon as square
        ('O1', {'type': 'type1', 'label': 'O'}),  # Oxygen as circle
        ('H1', {'type': 'type3', 'label': 'H'}),  # Hydrogens as triangles
        ('H2', {'type': 'type3', 'label': 'H'}),
        ('H3', {'type': 'type3', 'label': 'H'})
    ]
    G.add_nodes_from(atoms)
    
    # Add bonds as edges
    bonds = [
        ('C1', 'O1', {'type': 'black', 'label': 'double'}),
        ('C1', 'H1', {'type': 'white', 'label': 'single'}),
        ('C1', 'H2', {'type': 'white', 'label': 'single'}),
        ('C1', 'H3', {'type': 'white', 'label': 'single'})
    ]
    G.add_edges_from(bonds)
    
    return G

if __name__ == "__main__":
    # Create a molecule-like graph
    G = create_molecule_graph()
    
    # Create visualizer with default configurations
    visualizer = GraphVisualizer()
    
    # Try different layouts
    layouts = ['spring', 'circular', 'shell']
    
    for layout in layouts:
        # Draw and show the graph with different layouts
        visualizer.draw_graph(G, layout=layout, 
                            title=f"Molecule Graph with {layout.title()} Layout")
        visualizer.save(f"molecule_graph_{layout}.png")
        visualizer.show() 