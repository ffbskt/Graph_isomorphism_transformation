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

def diff_graphs(G1, G2):
    differences = {}

    # Check graph type
    if type(G1) != type(G2):
        differences["graph_type"] = (type(G1), type(G2))

    # Check node differences
    nodes_G1 = set(G1.nodes())
    nodes_G2 = set(G2.nodes())

    if nodes_G1 != nodes_G2:
        differences["missing_nodes"] = nodes_G1.symmetric_difference(nodes_G2)

    # Check node attribute differences
    node_attr_diffs = {}
    for node in nodes_G1 & nodes_G2:
        if G1.nodes[node] != G2.nodes[node]:
            node_attr_diffs[node] = (G1.nodes[node], G2.nodes[node])

    if node_attr_diffs:
        differences["node_attributes"] = node_attr_diffs

    # Check edge differences
    edges_G1 = set(G1.edges())
    edges_G2 = set(G2.edges())

    if edges_G1 != edges_G2:
        differences["missing_edges"] = edges_G1.symmetric_difference(edges_G2)

    # Check edge attribute differences
    edge_attr_diffs = {}
    for edge in edges_G1 & edges_G2:
        if G1[edge[0]][edge[1]] != G2[edge[0]][edge[1]]:
            edge_attr_diffs[edge] = (G1[edge[0]][edge[1]], G2[edge[0]][edge[1]])

    if edge_attr_diffs:
        differences["edge_attributes"] = edge_attr_diffs
    if differences != {}:
        print('differences', differences)
    return differences

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
    pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 'replacement', 'label': 'Re'})])

    # Apply transformation
    graph2_copy = graph2.copy()
    GC.clear()
    GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
    GC.transform(pattern)

    # Create expected result (matching the node IDs that are actually produced)
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (3, {'type': 'F', 'label': 'hb'}),  # The transformation will create this as node 3
        (1, {'type': 'b', 'label': 'b'}),
        (2, {'type': 'c', 'label': 'c'}),
    ])
    result_graph.add_edges_from([
        (1, 3, {'type': 1, 'label': '1'}),
        (3, 2, {'type': 1, 'label': '1'}),
    ])

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, GC.G, "Test 1: Single Node Replacement")

    # Compare graphs
    equal_nodes = set(GC.G.nodes()) == set(result_graph.nodes())
    equal_edges = set(GC.G.edges()) == set(result_graph.edges())
    
    # For debugging - check edges more carefully
    edge_data_match = True
    for u, v in result_graph.edges():
        if not GC.G.has_edge(u, v):
            print(f"Missing edge ({u}, {v}) in result")
            edge_data_match = False
        elif GC.G.get_edge_data(u, v) != result_graph.get_edge_data(u, v):
            print(f"Edge data mismatch for ({u}, {v}): Expected {result_graph.get_edge_data(u, v)}, Got {GC.G.get_edge_data(u, v)}")
            edge_data_match = False
    
    if not equal_nodes:
        print(f"Node mismatch. Expected: {result_graph.nodes(data=True)}, Got: {GC.G.nodes(data=True)}")
    if not equal_edges:
        print(f"Edge mismatch. Expected: {list(result_graph.edges(data=True))}, Got: {list(GC.G.edges(data=True))}")
        
    return equal_nodes and equal_edges and edge_data_match


def test_multiple_node_replacement(graph2=create_test_graph(type='linear'), visualize=False):
    """Test replacing a node with multiple connected nodes."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 'replacement', 'label': 'Re'}),
        (10, 12, {'type': 'replacement', 'label': 'Re'})
    ])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([(1, {'type': 'b', 'label': 'b'}), (2, {'type': 'c', 'label': 'c'}), (3, {'type': 'F', 'label': 'hb'}), (4, {'type': 'F', 'label': 'hb'})])
    result_graph.add_edges_from([(1, 3, {'type': 1, 'label': '1'}), (1, 4, {'type': 1, 'label': '1'}), (3, 4, {'type': 1, 'label': '1'}), (3, 2, {'type': 1, 'label': '1'}), (4, 2, {'type': 1, 'label': '1'})])

    # Apply transformation
    graph2_copy = graph2.copy()
    GC.clear()
    GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
    GC.transform(pattern)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, GC.G, "Test 2: Multiple Node Replacement")

    equals = diff_graphs(GC.G, result_graph) == {}
    return equals  #nx.utils.graphs_equal(result_graph, GC.G)


def test_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding a new node to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 1, 'label': '1'})])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([(1, {'type': 'b', 'label': 'b'}), 
                                 (2, {'type': 'c', 'label': 'c'}), 
                                 (3, {'type': 'F', 'label': 'hb'}), 
                                 (4, {'type': 'a', 'label': 'a'})]
                                )
    result_graph.add_edges_from([
         (1, 4, {'type': 1, 'label': '1'}), 
         (4, 3, {'type': 1, 'label': '1'}), 
         (4, 2, {'type': 1, 'label': '1'})
    ])

    # Apply transformation
    graph2_copy = graph2.copy()
    GC.clear()
    GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
    GC.transform(pattern)
    #print('-------in------G new ', GC.G.nodes(data=True), GC.G.edges)
    #print(result_graph.nodes(data=True), result_graph.edges)
    # print('-----------3--G new ', GC.G.nodes(data=True), GC.G.edges(data=True))
    # print('-----------3--result_graph ', result_graph.nodes(data=True), result_graph.edges(data=True))
    # print(diff_graphs(GC.G, result_graph))
    
    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, GC.G, "Test 3: Node Addition")

    equals = diff_graphs(GC.G, result_graph) == {} #nx.utils.graphs_equal(result_graph, GC.G)
    # print(f"Node Addition: {'PASSED' if equals else 'FAILED'}")
    return equals


def test_multiple_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding multiple connected nodes to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
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
    result_graph.add_nodes_from([(1, {'type': 'b', 'label': 'b'}), (2, {'type': 'c', 'label': 'c'}), (3, {'type': 'F', 'label': 'hb'}), (4, {'type': 'F', 'label': 'hb'}), (5, {'type': 'a', 'label': 'a'})])
    result_graph.add_edges_from([(1, 5, {'type': 1, 'label': '1'}), (3, 4, {'type': 1, 'label': '1'}), (5, 3, {'type': 1, 'label': '1'}), (5, 4, {'type': 1, 'label': '1'}), (5, 2, {'type': 1, 'label': '1'})])

    # Apply transformation
    graph2_copy = graph2.copy()
    GC.clear()
    GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
    GC.transform(pattern)

    # Visualize
    if visualize:
        VisG.visualize_transformation(graph2_copy, pattern, GC.G, "Test 4: Multiple Node Addition end")
    # print('-------------G new ', GC.G.nodes(data=True), GC.G.edges(data=True))
    # print('-------------result_graph ', result_graph.nodes(data=True), result_graph.edges(data=True))
    # print(diff_graphs(GC.G, result_graph))
    # 
    equals = diff_graphs(GC.G, result_graph) == {} #nx.utils.graphs_equal(result_graph, GC.G)
    return equals


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
    
    assert P_reindexed.nodes(data=True)[P_reindexed.graph['iB']]['type'] == 'base'
    assert P_reindexed.nodes(data=True)[P_reindexed.graph['iH']]['type'] == 'head'
    assert len(GC.G.nodes) == len(G.nodes) + len(pattern.nodes)
    # GC.add_label(G_reindexed, 'test')

    # print('\nPattern Base:')
    # print(f"Nodes: {list(P_reindexed.graph['pbase'].nodes(data=True))}")
    # print(f"Edges: {list(P_reindexed.graph['pbase'].edges(data=True))}")
    
    # print('\nPattern Head:')
    # print(f"Nodes: {list(P_reindexed.graph['phead'].nodes(data=True))}")
    # print(f"Edges: {list(P_reindexed.graph['phead'].edges(data=True))}")
    # print(P_reindexed.graph['iB'], P_reindexed.graph['iH'], P_reindexed.nodes(data=True))
    
    # print('\nGraph:')
    # print(f"Nodes: {list(G_reindexed.nodes(data=True))}")
    # print(f"Edges: {list(G_reindexed.edges(data=True))}")
    
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
    # test_GC()
    
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
    
    #VisG.visualize_transformation(G_copy, pattern, G, "Test 5: Multiple Node Addition end")







