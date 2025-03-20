import networkx as nx
import random
import numpy as np




def create_random_graph(num_nodes=2, num_edges=1, node_types=None, edge_types=None, node_labels=None, edge_labels=None):
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
        node_types = ['T']
    if node_labels is None:
        node_labels = ['a', 'b', 'c']
    if edge_types is None:
        edge_types = ['1',]
    if edge_labels is None:
        edge_labels = ['1']

    # Create a random graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        node_type = random.choice(node_types)
        node_label = random.choice(node_labels)
        G.add_node(i, **{'type': node_type, 'label': node_label})
    # Add edges
    for i in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        edge_type = random.choice(edge_types)
        edge_label = random.choice(edge_labels) 
        src_index = list(G.nodes())[src]
        dst_index = list(G.nodes())[dst]
        G.add_edge(src_index, dst_index, type=edge_type, label=edge_label)

    return G


def compose_pattern(phead, pbase, edges):
    """
    reolacement - should be parametr of edges between childs B and H
    !! correct pattern should have only all replacement edges from one node or all addedges..
    In other case we do not know remove or not gnode.
    """
    P = nx.compose(phead, pbase)
    P.add_edges_from(edges)
    P.add_node('B', type='base', label='B')
    for node in pbase.nodes():
        P.add_edge('B', node, type='hierarchy', label='1')
    P.add_node('H', type='head', label='H')
    for node in phead.nodes():
        P.add_edge('H', node, type='hierarchy', label='1')
    P.add_edge('B', 'H', type=1, label='1')
    # add in graph data pbase, phead, B index U index
    P.graph['pbase'] = pbase.copy()
    P.graph['phead'] = phead.copy()
    P.graph['iB'] = 'B'
    P.graph['iH'] = 'H'
    return P

def create_random_edge_list(base_nodes, head_nodes, num_connect_edges=None, edges_types=None):
    if edges_types is None:
        edges_types = ['replacement', '1']
    edges_types = np.random.choice(np.array(edges_types, dtype=object))
    if num_connect_edges is None:
        num_connect_edges = np.random.randint(1, len(head_nodes))
    edges = []
    for i in range(num_connect_edges):
        src = int(np.random.choice(list(base_nodes)))
        dst = int(np.random.choice(list(head_nodes)))
        edges.append((src, dst, {'type': edges_types, 'label': edges_types[:2]}))
    return edges


def create_random_pattern(num_phead_nodes=2, num_edges_head=2, num_pbase_nodes=1, num_edges_base=0, 
                         num_connect_edges=1, edges_types=['replacement', '1'],
                         node_types_base=None, edge_types_base=None, 
                         node_types_head=None, edge_types_head=None, 
                         node_labels_base=None, edge_labels_base=None, 
                         node_labels_head=None, edge_labels_head=None):
    """
    Create random pattern with random nodes and edges
    Args:
        num_phead_nodes (int): Number of nodes in the head
        num_pbase_nodes (int): Number of nodes in the base
        num_edges_head (int): Number of edges in the head
        num_edges_base (int): Number of edges in the base
        num_connect_edges (int): Number of connect edges
        node_types_base (list): List of node types
        edge_types_base (list): List of edge types
        node_types_head (list): List of node types
        edge_types_head (list): List of edge types
        node_labels_base (list): List of node labels of base pattern
        edge_labels_base (list): List of edge labels of base pattern
        node_labels_head (list): List of node labels of head pattern
        edge_labels_head (list): List of edge labels of head pattern
    Returns:
        nx.DiGraph: Random pattern
    """
    phead = create_random_graph(num_phead_nodes, num_edges_head, node_types_head, edge_types_head, node_labels_head, edge_labels_head)
    pbase = create_random_graph(num_pbase_nodes, num_edges_base, node_types_base, edge_types_base, node_labels_base, edge_labels_base)
    
    # Ensure phead node indices are greater than pbase
    max_pbase_index = max(pbase.nodes) if pbase.nodes else -1
    phead = nx.relabel_nodes(phead, lambda x: x + max_pbase_index + 1)
    
    edges = create_random_edge_list(pbase.nodes(), phead.nodes(), num_connect_edges, edges_types)
    return compose_pattern(phead, pbase, edges)


