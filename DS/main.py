import os
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import pandas as pd

from DS.Space.Collections import GraphCollection, NodeCollection
from DS.Space.Creations import compose_pattern
from DS.Visualisation.visg import VisG
from DS.Space.Creations import create_random_pattern, create_random_graph
from logging import getLogger

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
    graph = nx.DiGraph()
    if type == 'linear':
        graph.add_nodes_from([
            (0, {'type': 'a', 'label': 'a'}),
            (1, {'type': 'b', 'label': 'b'}),
            (2, {'type': 'c', 'label': 'c'}),
        ])
        graph.add_edges_from([
            (1, 0, {'type': 1, 'label': '1'}),
            (0, 2, {'type': 1, 'label': '1'}),
        ])
    elif type == 'triangle':
        graph.add_nodes_from([
            (0, {'type': 'a', 'label': 'a'}),
            (1, {'type': 'b', 'label': 'b'}),
            (2, {'type': 'c', 'label': 'c'}),
        ])
        graph.add_edges_from([
            (0, 1, {'type': 1, 'label': '1'}),
            (1, 2, {'type': 1, 'label': '1'}),
            (2, 0, {'type': 1, 'label': '1'}),
        ])
    elif type == 'double':
        graph.add_nodes_from([
            (0, {'type': 'a', 'label': 'a'}),
            (1, {'type': 'a', 'label': 'a'}),
            (2, {'type': 'c', 'label': 'c'}),
        ])
        graph.add_edges_from([
            (0, 1, {'type': 1, 'label': '1'}),
            (1, 2, {'type': 1, 'label': '1'}),
            (2, 0, {'type': 1, 'label': '1'}),
        ])

    return graph

#-------------------------------------------------------------------------

#import pandas as pd

class GraphTransformationInterface:
    def __init__(self, num_patterns=15, num_transformations=10):
        self.source_graph = create_random_graph(3, 2)
        self.target_graph = create_random_graph(10, 9)
        self.patterns = []
        for _ in range(num_patterns):
            num_pbase_nodes = 2 # random.randint(1, 3)
            num_phead_nodes = random.randint(1, 3)
            p = create_random_pattern(num_phead_nodes=num_phead_nodes, num_edges_head=num_phead_nodes - 1,
                                      num_pbase_nodes=num_pbase_nodes, num_edges_base=1,
                                      num_connect_edges=1, 
                                      node_types_base=None, edge_types_base=None, 
                                      node_types_head=None, edge_types_head=None, 
                                      node_labels_base=['a','b','c'], edge_labels_base=None, 
                                      node_labels_head=['a','b','c'], edge_labels_head=None)
            self.patterns.append(p)
            # print('pattern', p.nodes(data=True), p.edges())
        self.num_transformations = num_transformations
        self.transformation_results = []
        self.GC = GraphCollection()
    
    
    
    def transform_graph(self, graph, pattern, number_of_transformations=1):
        """Applies a transformation using the given pattern graph."""
        self.GC.clear()
        src = graph.copy()
        pat = pattern.copy()
        self.GC.add_graph_to_collection(src, label='source', is_pattern=False)
        #print(pat.nodes(), pat.edges())
        #self.GC.add_graph_to_collection(pat, label='pattern', is_pattern=True)
        self.GC.transform(pat, number_of_transformations=number_of_transformations)
        return self.GC.G
    
    def evaluate_similarity(self, graph):
        """Compares transformed graph with target graph, including node labels."""
        def get_label_pairs(G):
            label_pairs = set()
            for u, v in G.edges():
                label_pairs.add((G.nodes[u]['label'], G.nodes[v]['label']))
                # label_pairs.add((G.nodes[v]['label'], G.nodes[u]['label']))  # Ensure symmetry
            return label_pairs

        source_label_pairs = get_label_pairs(graph)
        target_label_pairs = get_label_pairs(self.target_graph)

        matched_pairs = len(source_label_pairs & target_label_pairs)

        return matched_pairs/len(target_label_pairs)
    
    def run_experiment(self):
        """Runs transformation experiment and logs results."""
        self.GC.get_graph_to_collection_log(self.source_graph, reindex_map=None, message='Source graph')
        self.GC.get_graph_to_collection_log(self.target_graph, reindex_map=None, message='Target graph')
        matched_pairs_default = self.evaluate_similarity(self.source_graph)
        for i, pattern in enumerate(self.patterns):
            transformed_graph = self.source_graph.copy()
            self.GC.get_graph_to_collection_log(pattern, reindex_map=None, message='Pattern graph')
            transformed_graph = self.transform_graph(transformed_graph, pattern, number_of_transformations=self.num_transformations)
            matched_pairs = self.evaluate_similarity(transformed_graph)
            self.transformation_results.append({
                #"Pattern": pattern,
                "Matched Pairs": round(matched_pairs, 2),
                "Default Matched Pairs": round(matched_pairs_default, 2)
                #"Total Pairs": total_pairs
            })
            #VisG.visualize_transformation(self.source_graph, pattern, transformed_graph, "Transformation " + str(i))
        VisG.visualize_transformation(self.source_graph, transformed_graph, self.target_graph, "Source, result, target end")
    
    def get_results(self):
        for i, result in enumerate(self.transformation_results):
            print(i, result)
        # return pd.DataFrame(self.transformation_results)
    
    def display_results(self):
        # df = self.get_results()
        # import ace_tools as tools
        # tools.display_dataframe_to_user("Transformation Results", df)
        print(self.transformation_results)




#-----------------------------------------------------------------------
def test_single_node_replacement_linear(graph2=create_test_graph(type='linear'), visualize=False):
    """Test replacing a single node with another node.
    if pattern type of edge is replacement,
    then we should not remove gnode and replace all edges in and out of gnode to hnode"""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'a', 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = compose_pattern(phead, pbase, [(10, 11, {'type': 'replacement', 'label': 'Re'})])

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
    pattern = compose_pattern(phead, pbase, [
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
    pbase.add_nodes_from([(10, {'type': 'a', 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
    pattern = compose_pattern(phead, pbase, [(10, 11, {'type': 1, 'label': '1'})])

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
    pattern = compose_pattern(phead, pbase, [
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

#-----------------------------------------------------------------------

def test_GC():
    GC.clear()
    print('\n--- Testing GraphCollection ---')
    G = create_test_graph(type='linear')
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'hb'}),
        (12, {'type': 'F', 'label': 'hb'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = compose_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (10, 12, {'type': 1, 'label': '1'})
    ])
    G_reindexed, G_reindex_map = GC.add_graph_to_collection(G, label='test', is_pattern=False)
    P_reindexed, P_reindex_map = GC.add_graph_to_collection(pattern, label='test_pattern', is_pattern=True)
    
    assert P_reindexed.nodes(data=True)[P_reindexed.graph['iB']]['type'] == 'base'
    assert P_reindexed.nodes(data=True)[P_reindexed.graph['iH']]['type'] == 'head'
    assert len(GC.G.nodes) == len(G.nodes) + len(pattern.nodes)
    
    return G_reindexed, P_reindexed


if __name__ == "__main__":
    # Run regular tests
    tests = [
        ("Single Node Replacement", test_single_node_replacement_linear),
        ("Multiple Node Replacement", test_multiple_node_replacement),
        ("Node Addition", test_node_addition),
        ("Multiple Node Addition", test_multiple_node_addition)
    ]

    # for test_name, test_func in tests:
    #     test_graph = create_test_graph(type='linear')
    #     result = test_func(test_graph, visualize=True)
    #     print(f"{test_name}: {'PASSED' if result else 'FAILED'}")

    # Run the GraphCollection test
    # test_GC()
    
    # Run a final transformation visualization
    G = create_test_graph(type='double')
    phead, pbase = nx.DiGraph(), nx.DiGraph()
    phead.add_nodes_from([
        (10, {'type': None, 'label': 'aa'}),
        (11, {'type': None, 'label': 'bb'}),
    ])
    phead.add_edges_from([(10, 11, {'type': 1, 'label': '1'})])
    pbase.add_nodes_from([(12, {'type': None, 'label': 'a'})])
    pattern = compose_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (12, 10, {'type': 'replacement', 'label': 'e'})
    ])
    G_copy = G.copy()
    # GC.clear()
    # GC.add_graph_to_collection(G, label='test', is_pattern=False)
    # GC.transform(pattern, number_of_transformations=2)
    #VisG.visualize_transformation(G_copy, pattern, GC.G, "Test 5: Multiple Node Addition")


    # Example Usage
    interface = GraphTransformationInterface(num_patterns=30, num_transformations=10)
    interface.run_experiment()
    #interface.display_results()
    interface.get_results()






