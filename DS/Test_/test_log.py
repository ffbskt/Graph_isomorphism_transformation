import unittest
import networkx as nx
import random
import os
from DS.Space.Collections import GraphCollection
from DS.Logger.Vis_last_log import LogReader
from DS.Logger.logger import JSONLogger

def create_random_graph(num_nodes=5):
    """Create a random graph with given number of nodes."""
    G = nx.DiGraph()
    
    # Add nodes with random attributes
    node_types = ['A', 'B', 'C']
    for i in range(num_nodes):
        G.add_node(i, type=random.choice(node_types), label=f'Node_{i}')
    
    # Add random edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < 0.3:  # 30% chance of edge
                G.add_edge(i, j, type='black', label='edge')
    
    return G

class TestLogging(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Clear any existing log file
        self.log_file = "Test_log_graph.json"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        
        # Initialize collections
        JSONLogger.reset_instance()
        logger = JSONLogger(filename=self.log_file)
        self.GC = GraphCollection(logger=logger)
        
    def test_graph_logging_and_reading(self):
        """Test that graphs added to collection are correctly logged and can be read back."""
        reindex_maps = []
        original_graphs = []
        
        # Create and add 3 random graphs
        for i in range(3):
            # Create random graph
            G = create_random_graph()
            original_graphs.append(G)
            G_reindexed, reindex_map = self.GC.add_graph_to_collection(G)
            reindex_maps.append(reindex_map)
        
        # Initialize visualizer and get last 3 logs
        visualizer = LogReader(self.log_file)
        last_logs = visualizer.get_last_n_logs()
        
        # Create graphs from logs
        graphs_from_log = visualizer.create_graphs_from_log(last_logs)
        
        # Verify each graph
        for i, (name, reconstructed_graph) in enumerate(graphs_from_log):
            original = original_graphs[i]
            reindex_map = reindex_maps[i]
            
            # Reindex original graph to match collection's indexing
            original = nx.relabel_nodes(original, reindex_map)
            
            
            result = nx.utils.graphs_equal(original, reconstructed_graph)
            
            if result is False:
                message = "Graphs do not match"
                # Compare number of nodes and edges
                nodes_match = len(original.nodes()) == len(reconstructed_graph.nodes())
                edges_match = len(original.edges()) == len(reconstructed_graph.edges())
            
                message += f"\nGraph {i}:"
                message += f"\nNodes match: {nodes_match} (Original: {len(original.nodes())}, Reconstructed: {len(reconstructed_graph.nodes())})"
                message += f"\nEdges match: {edges_match} (Original: {len(original.edges())}, Reconstructed: {len(reconstructed_graph.edges())})"
                
                # Compare node attributes
                node_attrs_match = all(
                    original.nodes[n] == reconstructed_graph.nodes[n]
                    for n in original.nodes()
                )
                message += f"\nNode attributes match: {node_attrs_match}"
                
                # Compare edge attributes
                edge_attrs_match = all(
                    original.edges[e] == reconstructed_graph.edges[e]
                    for e in original.edges()
                )
                message += f"\nEdge attributes match: {edge_attrs_match}"

                self.assertTrue(result, message)
        
    

if __name__ == "__main__":
    unittest.main()