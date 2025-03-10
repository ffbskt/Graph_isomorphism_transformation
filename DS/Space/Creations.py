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
    P.graph['pbase'] = pbase
    P.graph['phead'] = phead
    P.graph['iB'] = 'B'
    P.graph['iH'] = 'H'
    return P

def create_random_edge_list(base_nodes, head_nodes):
    label_type = np.random.choice(np.array(['replacement', '1'], dtype=object))
    number_of_edges = np.random.randint(1, len(head_nodes))
    edges = []
    for i in range(number_of_edges):
        src = np.random.choice(base_nodes)
        dst = np.random.choice(head_nodes)
        edges.append((src, dst, {'type': label_type}))
    return edges

def create_random_pattern(num_phead_nodes=2, num_pbase_nodes=1, num_edges_head=2, num_edges_base=0, node_types=None, edge_types=None, node_labels=None, edge_labels=None):
    phead = create_random_graph(num_phead_nodes, num_edges_head, node_types, edge_types, node_labels, edge_labels)
    pbase = create_random_graph(num_pbase_nodes, num_edges_base, node_types, edge_types, node_labels, edge_labels)
    
    # Ensure phead node indices are greater than pbase
    max_pbase_index = max(pbase.nodes) if pbase.nodes else -1
    phead = nx.relabel_nodes(phead, lambda x: x + max_pbase_index + 1)
    
    # Create random links between phead and pbase
    for _ in range(num_edges):
        u = random.choice(list(pbase.nodes))
        v = random.choice(list(phead.nodes))
        edge_type = random.choice(['replacement', '1'])
        edge_label = random.choice(['1'])
        phead.add_edge(u, v, type=edge_type, label=edge_label)

    edges = create_random_edge_list(pbase.nodes(), phead.nodes())
    return compose_pattern(phead, pbase, edges)

