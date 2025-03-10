import os
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from DS.Space.Collections import GraphCollection, NodeCollection
from DS.Space.Creations import compose_pattern
from DS.Visualisation.visg import VisG
from DS.Space.Creations import create_random_pattern, create_random_graph

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
    def __init__(self, num_patterns=5, num_transformations=10):
        self.source_graph = create_random_graph()
        self.target_graph = create_random_graph()
        self.patterns = [create_random_pattern() for _ in range(num_patterns)]
        self.num_transformations = num_transformations
        self.transformation_results = []
        self.GC = GraphCollection()
    
    
    
    def transform_graph(self, graph, pattern):
        """Applies a transformation using the given pattern graph."""
        self.GC.clear()
        src = graph.copy()
        pat = pattern.copy()
        self.GC.add_graph_to_collection(src, label='source', is_pattern=False)
        #self.GC.add_graph_to_collection(pat, label='pattern', is_pattern=True)
        self.GC.transform(pat, number_of_transformations=1)
        return self.GC.G
    
    def evaluate_similarity(self, graph):
        """Compares transformed graph with target graph, including node labels."""
        def get_label_pairs(G):
            label_pairs = set()
            for u, v in G.edges():
                label_pairs.add((G.nodes[u]['label'], G.nodes[v]['label']))
                label_pairs.add((G.nodes[v]['label'], G.nodes[u]['label']))  # Ensure symmetry
            return label_pairs

        source_label_pairs = get_label_pairs(graph)
        target_label_pairs = get_label_pairs(self.target_graph)

        matched_pairs = len(source_label_pairs & target_label_pairs)
        total_pairs = len(source_label_pairs)

        return matched_pairs, total_pairs
    
    def run_experiment(self):
        """Runs transformation experiment and logs results."""
        transformed_graph = self.source_graph.copy()
        for i, pattern in enumerate(self.patterns):

            transformed_graph = self.transform_graph(transformed_graph, pattern)
            matched_pairs, total_pairs = self.evaluate_similarity(transformed_graph)
            self.transformation_results.append({
                "Pattern": pattern,
                "Matched Pairs": matched_pairs,
                "Total Pairs": total_pairs
            })
            VisG.visualize_transformation(self.source_graph, pattern, transformed_graph, "Transformation " + str(i))
        VisG.visualize_transformation(self.source_graph, transformed_graph, self.target_graph, "Source, result, target end")
    
    def get_results(self):
        """Returns results as a pandas DataFrame."""
        
        return self.transformation_results
    
    def display_results(self):
        # df = self.get_results()
        # import ace_tools as tools
        # tools.display_dataframe_to_user("Transformation Results", df)
        print(self.transformation_results)

# Example Usage
interface = GraphTransformationInterface()
interface.run_experiment()
interface.display_results()


#-----------------------------------------------------------------------



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
    
    # Run the GraphCollection test
    test_GC()
    
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
    GC.clear()
    GC.add_graph_to_collection(G, label='test', is_pattern=False)

    GC.transform(pattern, number_of_transformations=2)
    #print('GC.G', GC.G.nodes(data=True), GC.G.edges(data=True))
    
    #VisG.visualize_transformation(G_copy, pattern, GC.G, "Test 5: Multiple Node Addition end")







