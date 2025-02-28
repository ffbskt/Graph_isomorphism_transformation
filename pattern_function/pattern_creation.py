import sys
import random
#sys.path.append(r'C:\Users\Denis\PycharmProjects\Graph_isomorphism_transformation\visg')
from visg import VisG
import networkx as nx
import matplotlib.pyplot as plt

def create_random_graph(num_nodes, num_edges, node_types=None, edge_types=None, node_labels=None, edge_labels=None):
    """
    Create a random graph with specified number of nodes and edges, with random types and labels
    
    Args:
        num_nodes (int): Number of nodes
        num_edges (int): Number of edges
        node_types (list): List of possible node types/labels
        edge_types (list): List of possible edge types/labels
        node_labels (list): List of possible node labels
        edge_labels (list): List of possible edge labels
    Returns:
        dict: Graph representation with nodes and edges
    """
    if node_types is None:
        node_types = ['a', 'b', 'c']
    if node_labels is None:
        node_labels = ['a', 'b', 'c']
    if edge_types is None:
        edge_types = [1, 2, 3]
    if node_labels is None:
        node_labels = ['a', 'b', 'c']

    # Create a random graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        node_type = random.choice(node_types)
        node_label = random.choice(node_labels)
        G.add_node(i, type=node_type, label=node_label) 

    # Add edges
    for i in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        edge_type = random.choice(edge_types)
        edge_label = random.choice(edge_labels) 
        G.add_edge(src, dst, type=edge_type, label=edge_label)

    return G


def find_isomorphisms(pattern_base, G_text, node_match=None, edge_match=None):
    return nx.algorithms.isomorphism.GraphMatcher(
        G_text, pattern_base, node_match=node_match, edge_match=edge_match).subgraph_isomorphisms_iter()



def create_pattern(phead, pbase, edges):
    P = nx.compose(phead, pbase)
    P.add_edges_from(edges)
    return P




















if __name__ == "__main__":
    # Create a random graph
    num_nodes = 5
    num_edges = 7
    node_types = ['a', 'b', 'c']
    node_labels = ['a', 'b', 'c']
    edge_labels = ['1', '2', '3']
    edge_types = [1, 2, 3]
    
    graph1 = create_random_graph(num_nodes, num_edges, node_types, edge_types, node_labels, edge_labels)
    print("Generated Graph 1:")
    print(graph1)
    
    # Create pattern object
    phead, pbase = nx.DiGraph(), nx.DiGraph()
    phead.add_nodes_from([(0, {'type': 'a', 'label': 'a'}),])
    pbase.add_nodes_from([(1, {'type': 'a', 'label': 'a'}),
                          (2, {'type': 'a', 'label': 'b'}),
                          ])
    pbase.add_edges_from([(1, 2),])
    pattern = create_pattern(phead, pbase, [(0, 1, {'type': 1, 'label': '1'}), (1, 2, {'type': 2, 'label': '2'})])
    print(pattern)

    # Create figure with two subplots side by side
    plt.figure(figsize=(12, 5))
    
    # Left subplot for graph1
    plt.subplot(121)
    vis1 = VisG()
    vis1.add_graph(graph1) 
    vis1.draw(layout='spring', title='Random Graph')
    
    # Right subplot for pattern
    plt.subplot(122)
    vis2 = VisG()
    vis2.add_graph(pattern)
    vis2.draw(layout='spring', title='Pattern Graph')
    
    plt.tight_layout()
    plt.show()

    # find isomorphisms
    isomorphisms = find_isomorphisms(pattern, graph1)
    print(isomorphisms)

