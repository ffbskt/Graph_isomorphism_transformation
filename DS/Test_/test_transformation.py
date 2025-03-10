import unittest
import networkx as nx
import random
from DS.Space.Collections import GraphCollection, NodeCollection
from DS.Space.Creations import compose_pattern
from DS.Visualisation.visg import VisG
from DS.main import create_test_graph, diff_graphs

# Create collections as in main.py
NC = NodeCollection()
GC = GraphCollection(NC=NC)

class TestTransformation(unittest.TestCase):
    
    def setUp(self):
        # Reset the graph collection before each test
        GC.clear()
    
    def test_single_node_replacement_linear(self):
        """Test replacing a single node with another node.
        if pattern type of edge is replacement,
        then we should not remove gnode and replace all edges in and out of gnode to hnode"""
        # Create test graph
        graph2 = create_test_graph(type='linear')
        
        # Create pattern
        pbase, phead = nx.DiGraph(), nx.DiGraph()
        pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
        phead.add_nodes_from([(11, {'type': 'F', 'label': 'hb'})])
        pattern = compose_pattern(phead, pbase, [(10, 11, {'type': 'replacement', 'label': 'Re'})])

        # Apply transformation
        graph2_copy = graph2.copy()
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
        
        self.assertTrue(equal_nodes, f"Node mismatch. Expected: {result_graph.nodes(data=True)}, Got: {GC.G.nodes(data=True)}")
        self.assertTrue(equal_edges, f"Edge mismatch. Expected: {list(result_graph.edges(data=True))}, Got: {list(GC.G.edges(data=True))}")
        self.assertTrue(edge_data_match, "Edge data mismatch")

    def test_multiple_node_replacement(self):
        """Test replacing a node with multiple connected nodes."""
        # Create test graph
        graph2 = create_test_graph(type='linear')
        
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
        GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
        GC.transform(pattern)

        # Verify result
        self.assertEqual({}, diff_graphs(GC.G, result_graph), "Graphs should be identical")

    def test_node_addition(self):
        """Test adding a new node to existing node."""
        # Create test graph
        graph2 = create_test_graph(type='linear')
        
        # Create pattern
        pbase, phead = nx.DiGraph(), nx.DiGraph()
        pbase.add_nodes_from([(10, {'type': None, 'label': 'a'})])
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
        GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
        GC.transform(pattern)
        
        # Verify result
        self.assertEqual({}, diff_graphs(GC.G, result_graph), "Graphs should be identical")

    def test_multiple_node_addition(self):
        """Test adding multiple connected nodes to existing node."""
        # Create test graph
        graph2 = create_test_graph(type='linear')
        
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
        GC.add_graph_to_collection(graph2, label='test', is_pattern=False)
        GC.transform(pattern)

        # Verify result
        self.assertEqual({}, diff_graphs(GC.G, result_graph), "Graphs should be identical")


if __name__ == "__main__":
    unittest.main() 