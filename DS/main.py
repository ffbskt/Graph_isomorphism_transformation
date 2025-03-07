import os
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from DS.Space.Collections import GraphCollection, NodeCollection
from DS.Space.Creations import create_pattern
from DS.Visualisation.visg import VisG

random.seed(1)
NC = NodeCollection()
GC = GraphCollection(NC=NC)


class Transformation:
    """Class to handle graph transformations with patterns"""
    
    def __init__(self, pattern):
        """
        Initialize transformation with a pattern
        
        Args:
            pattern (nx.DiGraph): Pattern graph with base and head
        """
        self.pattern = pattern
        self.pbase = pattern.graph.get('pbase', None)
        self.phead = pattern.graph.get('phead', None)
        self.iB = pattern.graph.get('iB', 'B')
        self.iH = pattern.graph.get('iH', 'H')
        
    def transform(self, G, visualize=False):
        """
        Apply the transformation pattern to graph G
        
        Args:
            G (nx.DiGraph): Graph to transform
            visualize (bool): Whether to visualize the transformation
            
        Returns:
            nx.DiGraph: Transformed graph
        """
        G_copy = G.copy()
        apply_pattern(self.pattern, G)
        
        if visualize:
            VisG.visualize_transformation(G_copy, self.pattern, G, "Graph Transformation")
            
        return G


def find_isomorphisms(pattern_base, G, node_match=None, edge_match=None):
    return nx.algorithms.isomorphism.DiGraphMatcher(G, pattern_base, node_match=node_match, edge_match=edge_match).subgraph_isomorphisms_iter()


def replace_edges(G, gnode, hnode, except_edges={}):
    # replace all edges in and out of gnode to hnode
    # get edge list from gnode and in gnode
    edges_out = list(G.edges(gnode, data=True))
    edges_in = list(G.in_edges(gnode, data=True))
    for edge in edges_out:
        if edge[2]['type'] not in except_edges:
            G.add_edge(hnode, edge[1], **edge[2])
    for edge in edges_in:
        if edge[2]['type'] not in except_edges:
            G.add_edge(edge[0], hnode, **edge[2])


def replace_by_isomorphism(pattern, G, iso):
    """
    iso is dict of graph node -> pattern base node
    """
    # get pattern head and remove 'h' and 'b'
    # add head and  base and head to G then remove base.///////////////// TODO
    G.add_nodes_from(pattern.nodes(data=True))
    G.add_edges_from(pattern.edges(data=True))

    for gnode in iso:
        bnode = iso[gnode]
        # first look at nodes from base that link with head.
        replace = False
        for edge in pattern.edges(bnode):
            # if edge type 'replacement'
            hnode = edge[1]
            if pattern.edges[edge]['type'] == 'replacement':
                replace_edges(G, gnode, hnode)  # replace all edges in and out of gnode to hnode
                replace = True
            else:  # add hnode
                replace_edges(G, bnode, gnode, except_edges={'replacement',
                                                             'hierarchy'})  # replace all edges in and out of bnode to gnode
        if replace:
            G.remove_node(gnode)
    
    # Get the base pattern nodes to remove
    pbase = GC.subgraph_with_neighbors(node_list=['B'], G=pattern, depth=1)
    nodes_to_remove = list(pbase.nodes())
    G.remove_nodes_from(nodes_to_remove)


def apply_pattern(pattern, G):
    """
    Apply pattern to G
    """
    # Get the base pattern by getting subgraph around node 'B'
    pbase = GC.subgraph_with_neighbors(node_list=['B'], G=pattern, depth=1, remove_nodes=['B', 'H'])
    # find L-P
    # print('pbase', pbase.nodes(data=True), pbase.edges(data=True))
    for iso in list(find_isomorphisms(pbase, G)):
        # print('iso', iso, G.nodes(data=True), G.edges(data=True))
        replace_by_isomorphism(pattern, G, iso)


def create_test_graph(type='linear'):
    # Create test graph
    if type == 'linear':
        graph2 = nx.DiGraph()
        graph2.add_nodes_from([
            (0, {'type': 'a', 'label': 'a'}),
            (1, {'type': 'b', 'label': 'b'}),
            (2, {'type': 'c', 'label': 'c'}),
        ])
        graph2.add_edges_from([
            (1, 0, {'type': 1, 'label': '1'}),
            (0, 2, {'type': 1, 'label': '1'}),
        ])
    elif type == 'triangle':
        graph2.add_nodes_from([
            (0, {'type': 'a', 'label': 'a'}),
            (1, {'type': 'b', 'label': 'b'}),
            (2, {'type': 'c', 'label': 'c'}),
        ])
        graph2.add_edges_from([
            (0, 1, {'type': 1, 'label': '1'}),
            (1, 2, {'type': 1, 'label': '1'}),
            (2, 0, {'type': 1, 'label': '1'}),
        ])
    return graph2


def test_single_node_replacement_linear(graph2=create_test_graph(type='linear'), visualize=False):
    """Test replacing a single node with another node.
    if pattern type of edge is replacement,
    then we should not remove gnode and replace all edges in and out of gnode to hnode"""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 'replacement', 'label': '1'})])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (1, {'type': 'b', 'label': 'b'}),
        (2, {'type': 'c', 'label': 'c'}),
    ])
    result_graph.add_edges_from([
        (1, 11, {'type': 1, 'label': '1'}),
        (11, 2, {'type': 1, 'label': '1'}),
    ])

    # Apply transformation
    graph2_copy = graph2.copy()
    iso = list(find_isomorphisms(pbase, graph2))[0]
    replace_by_isomorphism(pattern, graph2, iso)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, graph2, "Test 1: Single Node Replacement")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_multiple_node_replacement(graph2=create_test_graph(type='linear'), visualize=False):
    """Test replacing a node with multiple connected nodes."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 'replacement', 'label': '1'}),
        (10, 12, {'type': 'replacement', 'label': '1'})
    ])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
        (1, {'type': 'b', 'label': 'b'}),
        (2, {'type': 'c', 'label': 'c'}),
    ])
    result_graph.add_edges_from([
        (1, 11, {'type': 1, 'label': '1'}),
        (11, 2, {'type': 1, 'label': '1'}),
        (12, 2, {'type': 1, 'label': '1'}),
        (1, 12, {'type': 1, 'label': '1'}),
        (11, 12, {'type': 1, 'label': '1'}),
    ])

    # Apply transformation
    graph2_copy = graph2.copy()
    iso = list(find_isomorphisms(pbase, graph2))[0]
    replace_by_isomorphism(pattern, graph2, iso)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, graph2, "Test 2: Multiple Node Replacement")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding a new node to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 1, 'label': '1'})])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (0, {'type': 'a', 'label': 'a'}),
        (1, {'type': 'b', 'label': 'b'}),
        (2, {'type': 'c', 'label': 'c'}),
    ])
    result_graph.add_edges_from([
        (0, 11, {'type': 1, 'label': '1'}),
        (1, 0, {'type': 1, 'label': '1'}),
        (0, 2, {'type': 1, 'label': '1'}),
    ])

    # Apply transformation
    graph2_copy = graph2.copy()
    iso = list(find_isomorphisms(pbase, graph2))[0]
    replace_by_isomorphism(pattern, graph2, iso)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, graph2, "Test 3: Node Addition")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_multiple_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding multiple connected nodes to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (10, 12, {'type': 1, 'label': '1'})
    ])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
        (0, {'type': 'a', 'label': 'a'}),
        (1, {'type': 'b', 'label': 'b'}),
        (2, {'type': 'c', 'label': 'c'}),
    ])
    result_graph.add_edges_from([
        (0, 11, {'type': 1, 'label': '1'}),
        (0, 12, {'type': 1, 'label': '1'}),
        (1, 0, {'type': 1, 'label': '1'}),
        (0, 2, {'type': 1, 'label': '1'}),
        (11, 12, {'type': 1, 'label': '1'})
    ])

    # Apply transformation
    graph2_copy = graph2.copy()
    iso = list(find_isomorphisms(pbase, graph2))[0]
    replace_by_isomorphism(pattern, graph2, iso)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, graph2, "Test 4: Multiple Node Addition")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_GC():
    print('\n--- Testing GraphCollection ---')
    G = create_test_graph(type='linear')
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (10, 12, {'type': 1, 'label': '1'})
    ])
    G_reindexed, G_reindex_map = GC.add_graph_to_collection(G, label='test', is_pattern=False)
    P_reindexed, P_reindex_map = GC.add_graph_to_collection(pattern, label='test_pattern', is_pattern=True)
    
    print('\nPattern Base:')
    print(f"Nodes: {list(P_reindexed.graph['pbase'].nodes(data=True))}")
    print(f"Edges: {list(P_reindexed.graph['pbase'].edges(data=True))}")
    
    print('\nPattern Head:')
    print(f"Nodes: {list(P_reindexed.graph['phead'].nodes(data=True))}")
    print(f"Edges: {list(P_reindexed.graph['phead'].edges(data=True))}")
    print(P_reindexed.graph['iB'], P_reindexed.graph['iH'], P_reindexed.nodes(data=True))
    
    print('\nGraph:')
    print(f"Nodes: {list(G_reindexed.nodes(data=True))}")
    print(f"Edges: {list(G_reindexed.edges(data=True))}")
    
    return G_reindexed, P_reindexed


if __name__ == "__main__":
    # Run regular tests
    tests = [
        ("Single Node Replacement", test_single_node_replacement_linear),
        ("Multiple Node Replacement", test_multiple_node_replacement),
        ("Node Addition", test_node_addition),
        ("Multiple Node Addition", test_multiple_node_addition)
    ]

    for test_name, test_func in tests:
        test_graph = create_test_graph(type='linear')
        result = test_func(test_graph, visualize=True)
        print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
    
    # Run the GraphCollection test
    test_GC()
    
    # Run a final transformation visualization
    G = create_test_graph(type='linear')
    phead, pbase = nx.DiGraph(), nx.DiGraph()
    phead.add_nodes_from([
        (10, {'type': 'T', 'label': 'aa'}),
        (11, {'type': 'F', 'label': 'bb'}),
    ])
    phead.add_edges_from([(10, 11, {'type': 1, 'label': '1'})])
    pbase.add_nodes_from([(12, {'type': 'T', 'label': 'a'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (12, 10, {'type': 1, 'label': 'e'})
    ])
    G_copy = G.copy()
    apply_pattern(pattern, G)
    VisG.visualize_transformation(G_copy, pattern, G, "Test 5: Multiple Node Addition end")







