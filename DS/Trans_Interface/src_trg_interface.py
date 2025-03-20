from DS.Space.Creations import create_random_pattern, create_random_graph
from logging import getLogger
import random
import numpy as np
import networkx as nx
import pandas as pd
from DS.Space.Collections import GraphCollection
from DS.Visualisation.visg import VisG
from collections import Counter




class GraphTransformationInterface:
    def __init__(self, num_patterns=15, num_transformations=10, node_labels=[0,1], edge_types=['1', 'replacement']):
        self.source_graph = create_random_graph(3, 2, node_labels=node_labels, edge_types=edge_types)
        self.target_graph = create_random_graph(10, 9, node_labels=node_labels, edge_types=edge_types)
        self.patterns = []
        for _ in range(num_patterns):
            num_pbase_nodes = 2 # random.randint(1, 3)
            num_phead_nodes = random.randint(1, 3)
            p = create_random_pattern(num_phead_nodes=num_phead_nodes, num_edges_head=num_phead_nodes - 1,
                                      num_pbase_nodes=num_pbase_nodes, num_edges_base=1,
                                      num_connect_edges=1, 
                                      node_types_base=None, edge_types_base=edge_types, 
                                      node_types_head=None, edge_types_head=edge_types, 
                                      node_labels_base=node_labels, edge_labels_base=None, 
                                      node_labels_head=node_labels, edge_labels_head=None)
            self.patterns.append(p)
            # print('pattern', p.nodes(data=True), p.edges())
        self.num_transformations = num_transformations
        self.transformation_results = []
        self.GC = GraphCollection()
    
    
    
    def transform_graph(self, pattern, number_of_transformations=1):
        """Applies a transformation using the given pattern graph."""
        pat = pattern.copy()
        self.GC.transform(pat, number_of_transformations=number_of_transformations)
        return self.GC.G
    
    def re_init(self):
        self.GC.clear()
        self.GC.add_graph_to_collection(self.source_graph.copy(), label='transformed', is_pattern=False)
    
    
    def manual_experiment(self, epoches=10, steps=10): 
        """Runs transformation experiment and logs results."""
        self.GC.get_graph_to_collection_log(self.source_graph, reindex_map=None, message='Source graph')
        self.GC.get_graph_to_collection_log(self.target_graph, reindex_map=None, message='Target graph')
        matched_pairs_default = self.evaluate_similarity(self.source_graph)
        for epoch in range(epoches):
            self.re_init()
            for step in range(steps):
                i = np.random.randint(0, len(self.patterns))
                self.apply_pattern(i)
            matched_pairs = self.evaluate_similarity(self.GC.G)
            self.transformation_results.append({
                "Epoch": epoch,
                "Matched Pairs": round(matched_pairs, 2),
                "Default Matched Pairs": round(matched_pairs_default, 2)
                #"Total Pairs": total_pairs
            })
            #VisG.visualize_transformation(self.source_graph, pattern, transformed_graph, "Transformation " + str(i))
        VisG.visualize_transformation(self.source_graph, self.GC.G, self.target_graph, "Source, result, target end")
    
    def evaluate_similarity(self, graph):
        pair_set_similarity = self._label_pair_frequency_similarity(graph)
        return pair_set_similarity
        # edit_distance = self.edit_distance_similarity(graph)
        # return edit_distance

    

    

    def _label_pair_frequency_similarity(self, graph):
        """Compares frequency of label pairs in edges between graphs."""

        def get_label_pair_counts(G):
            counts = Counter()
            for u, v in G.edges():
                pair = (G.nodes[u]['label'], G.nodes[v]['label'])
                counts[pair] += 1
            return counts

        source_counts = get_label_pair_counts(graph)
        target_counts = get_label_pair_counts(self.target_graph)

        all_pairs = set(source_counts.keys()).union(target_counts.keys())
        total_diff = sum(abs(source_counts[pair] - target_counts[pair]) for pair in all_pairs)
        total_edges = sum(target_counts.values())

        similarity = 1 - (total_diff / (2 * total_edges)) if total_edges else 1.0

        return similarity



    def edit_distance_similarity(self, graph):
        """Normalized graph edit distance similarity measure."""
        def node_subst_cost(n1, n2):
            return 0 if n1['label'] == n2['label'] else 1
        
        ged = nx.graph_edit_distance(graph, self.target_graph,
                                    node_subst_cost=node_subst_cost,
                                    timeout=0.5)  # limit to 0.5s to avoid excessive computation
        
        if ged is None:
            return 0.0  # timed out, assume worst similarity
        
        max_possible_distance = max(graph.number_of_nodes() + graph.number_of_edges(),
                                    self.target_graph.number_of_nodes() + self.target_graph.number_of_edges())
        similarity = 1 - (ged / max_possible_distance)
        return max(0.0, similarity)  # ensure non-negative


    def get_cur_score(self):
        return self.evaluate_similarity(self.GC.G)

    def get_target(self):
        return self.target_graph

    def get_current_G(self):
        return self.GC.G

    def apply_pattern(self, i):
        self.transform_graph(self.patterns[i], number_of_transformations=1)
    
    def get_results(self):
        for i, result in enumerate(self.transformation_results):
            print(i, result)
        # return pd.DataFrame(self.transformation_results)

    def get_number_of_patterns(self):
        return len(self.patterns)
    
    def display_results(self):
        # df = self.get_results()
        # import ace_tools as tools
        # tools.display_dataframe_to_user("Transformation Results", df)
        print(self.transformation_results)

    def print_graps(self):
        print('src: ', [self.source_graph.nodes[i]['label'] for i in self.source_graph.nodes()], self.source_graph.edges())
        print('target: ', [self.target_graph.nodes[i]['label'] for i in self.target_graph.nodes()], self.target_graph.edges())
        for p in self.patterns:
            mp = {1:'1', '1': '1', 'replacement': 'r', 'hierarchy': 'h'}
            print('pattern: ', [p.nodes[i]['label'] for i in p.nodes()], [(i, mp[p.edges()[i]['type']]) for i in p.edges()])