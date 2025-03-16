import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Dict, Set, Callable, Any, Optional
import random
random.seed(42)
np.random.seed(42)

class McSplit:
    def __init__(self, G1: nx.DiGraph, G2: nx.DiGraph, 
                 node_match: Optional[Callable[[Dict, Dict], bool]] = None,
                 edge_match: Optional[Callable[[Dict, Dict], bool]] = None):
        """
        Initialize McSplit for directed graphs.

        Args:
            G1: First directed graph (networkx.DiGraph)
            G2: Second directed graph (networkx.DiGraph)
            node_match: Function to compare node attributes (default: None)
            edge_match: Function to compare edge attributes (default: None)
        """
        self.G1 = G1
        self.G2 = G2
        self.node_match = node_match or (lambda n1, n2: True)
        self.edge_match = edge_match or (lambda e1, e2: True)
        self.best_solution = {}  # Store best node mapping
        self.max_size = 0  # Maximum common subgraph size found

    def is_valid_mapping(self, mapping: Dict[Any, Any], u: Any, v: Any) -> bool:
        """
        Check if adding node pair (u,v) to the current mapping maintains graph structure.
        
        Args:
            mapping: Current node mapping from G1 to G2
            u: Node from G1 to add
            v: Node from G2 to add
            
        Returns:
            bool: True if adding (u,v) maintains isomorphism
        """
        # Check node attributes match
        if not self.node_match(self.G1.nodes[u], self.G2.nodes[v]):
            return False

        # Check edges with existing mapping
        for u1, v1 in mapping.items():
            # Check edges u -> u1 and v -> v1
            if self.G1.has_edge(u, u1) != self.G2.has_edge(v, v1):
                return False
            if self.G1.has_edge(u, u1) and self.G2.has_edge(v, v1):
                if not self.edge_match(self.G1[u][u1], self.G2[v][v1]):
                    return False
                    
            # Check edges u1 -> u and v1 -> v
            if self.G1.has_edge(u1, u) != self.G2.has_edge(v1, v):
                return False
            if self.G1.has_edge(u1, u) and self.G2.has_edge(v1, v):
                if not self.edge_match(self.G1[u1][u], self.G2[v1][v]):
                    return False
        return True

    def mcs(self, mapping: Dict[Any, Any], candidates_G1: Set[Any], candidates_G2: Set[Any]):
        """
        Recursive function to find the Maximum Common Subgraph.
        
        Args:
            mapping: Current node mapping from G1 to G2
            candidates_G1: Remaining candidates in G1
            candidates_G2: Remaining candidates in G2
        """
        # Update best solution if we found a larger common subgraph
        if len(mapping) > self.max_size:
            self.best_solution = mapping.copy()
            self.max_size = len(mapping)

        if not candidates_G1 or not candidates_G2:
            return

        # Select a node from G1 with highest degree as pivot
        u = max(candidates_G1, key=lambda x: self.G1.degree(x), default=None)
        if u is None:
            return

        # Try mapping u to each remaining candidate in G2
        for v in candidates_G2:
            if self.is_valid_mapping(mapping, u, v):
                new_mapping = mapping.copy()
                new_mapping[u] = v
                
                # Update candidate sets
                new_candidates_G1 = candidates_G1 - {u}
                new_candidates_G2 = candidates_G2 - {v}
                
                self.mcs(new_mapping, new_candidates_G1, new_candidates_G2)

    def find_mcs(self) -> Dict[Any, Any]:
        """
        Entry point to compute the Maximum Common Directed Subgraph.
        
        Returns:
            Dict[Any, Any]: Mapping from nodes in G1 to nodes in G2 representing the MCS
        """
        self.mcs({}, set(self.G1.nodes), set(self.G2.nodes))
        return self.best_solution





def node_match_by_type_label(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    """
    Match nodes based on their type and label attributes.
    
    Args:
        n1: Node attributes from first graph
        n2: Node attributes from second graph
        
    Returns:
        bool: True if nodes match
    """
    return n1.get('type') == n2.get('type') and n1.get('label') == n2.get('label')

def edge_match_by_type(e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
    """
    Match edges based on their type attribute.
    
    Args:
        e1: Edge attributes from first graph
        e2: Edge attributes from second graph
        
    Returns:
        bool: True if edges match
    """
    return e1.get('type') == e2.get('type')



def grow_graph(G: nx.DiGraph, num_connections: int, size_of_add_part: int, base_offset: int = 100):
    """
    Grow graph while preserving isomorphism by ensuring consistent node indexing and edge additions.

    :param G: Base directed graph to expand
    :param num_connections: Number of connections between the old and new parts
    :param size_of_add_part: Size of the new graph part to be added
    :param base_offset: Starting node ID for new nodes (should be consistent across calls)
    :return: A new expanded graph maintaining isomorphism
    """
    # Clone the original graph to prevent in-place modifications
    G_new = G.copy()
    
    max_node_id = max(G_new.nodes)
    
    # Generate a random subgraph of size `size_of_add_part`
    G_add = nx.gnp_random_graph(size_of_add_part, 0.5, directed=True)
    
    # Consistently relabel nodes in the new graph (use offset to ensure same renaming in both calls)
    relabel_map = {i: i + base_offset for i in G_add.nodes}
    G_add = nx.relabel_nodes(G_add, relabel_map)
    
    # Add nodes and edges to G_new
    G_new.add_nodes_from(G_add.nodes(data=True))  # Preserve attributes
    G_new.add_edges_from(G_add.edges(data=True))  # Preserve edge attributes
    
    # Maintain isomorphism by adding the same connections in a structured way
    rng = np.random.default_rng(42)  # Use a fixed seed for reproducibility
    old_nodes = list(G_new.nodes)[:max_node_id + 1]  # Select only old nodes
    
    for _ in range(num_connections):
        src = rng.choice(old_nodes)  # Pick a consistent old node
        dst = rng.choice(list(G_add.nodes))  # Pick from new nodes
        G_new.add_edge(int(src), int(dst))
    
    return G_new



def check_solution(G1: nx.DiGraph, G2: nx.DiGraph, common_graph: nx.DiGraph, solution: nx.DiGraph, mapping: Dict[Any, Any]):
    """
    Check if the common_graph is isomorphic to both G1 and G2.

    :param G1: First directed graph (DiGraph)
    :param G2: Second directed graph (DiGraph)
    :param common_graph: Proposed maximum common subgraph (DiGraph)
    :return: True if common_graph is isomorphic to both G1 and G2, False otherwise
    """
        
    #if len(mapping) < len(common_graph.nodes):
    #    return False
    
    # Check if common_graph is isomorphic to G1
    GM1 = DiGraphMatcher(G1, solution)
    is_sub_iso_G1 = GM1.subgraph_is_isomorphic()

    # Check if common_graph is isomorphic to G2
    GM2 = DiGraphMatcher(G2, solution)
    is_sub_iso_G2 = GM2.subgraph_is_isomorphic()

    GM22 = DiGraphMatcher(G2, common_graph)
    is_sub_iso_G22 = GM22.subgraph_is_isomorphic()


    
    # Return True only if common_graph is isomorphic to both G1 and G2
    #print('solution', solution.edges())
    #print(common_graph.edges())
    verify_subgraph_isomorphism_manual(G1, solution, mapping)
    verify_subgraph_isomorphism_manual(G2, solution, mapping)
    default_mapping = {i: i for i in common_graph.nodes}
    verify_subgraph_isomorphism_manual(G1, common_graph, default_mapping)

    return (is_sub_iso_G1 and is_sub_iso_G2 and is_sub_iso_G22, 
            len(common_graph), len(solution), 
            len(list(GM2.subgraph_isomorphisms_iter())), len(list(GM1.subgraph_isomorphisms_iter())),
            mapping
            )


def edge_node_equal(G1, G2):
    nodes_equal = set(G1.nodes) - set(G2.nodes)
    edges_equal = set(G1.edges) - set(G2.edges)
    return nodes_equal, edges_equal


def test_grow_graph(comon_G, G1, G2):
    GM1 = DiGraphMatcher(G1, comon_G)
    is_sub_iso_G1 = GM1.subgraph_is_isomorphic()

    # Check if common_graph is isomorphic to G2
    GM2 = DiGraphMatcher(G2, comon_G)
    is_sub_iso_G2 = GM2.subgraph_is_isomorphic()

    # check if nodes are equal and edges are equal
    nodes_equal, edges_equal = edge_node_equal(comon_G, G2)
    nodes_equal1, edges_equal1 = edge_node_equal(comon_G, G1)
    if is_sub_iso_G1 and is_sub_iso_G2:
        return True
    print(is_sub_iso_G1, is_sub_iso_G2, nodes_equal, edges_equal, nodes_equal1, edges_equal1, list(GM2.isomorphisms_iter()), list(GM1.isomorphisms_iter()))
    return False

     
def verify_subgraph_isomorphism_manual(G1, G2, mapping):
    """
    Verify if the provided mapping is a valid subgraph isomorphism from G2 to G1.

    Parameters:
    - G1: The target graph (networkx.Graph) (should be larger or equal in structure)
    - G2: The subgraph to be matched (networkx.Graph) (should be smaller or equal)
    - mapping: A dictionary {node_in_G2: node_in_G1} representing the node correspondence

    Returns:
    - bool: True if the mapping is a valid subgraph isomorphism, False otherwise
    """
    
    # Step 1: Check if all mapped nodes exist in G1
    for node_G2, mapped_node_G1 in mapping.items():
        if mapped_node_G1 not in G1:
            print(f"Node {mapped_node_G1} in G1 is missing (expected from G2's node {node_G2}).")
            return False  # A required node does not exist in G1

    # Step 2: Check if all edges in G2 exist in G1 after mapping
    for u, v in G2.edges():
        mapped_u, mapped_v = mapping[u], mapping[v]
        if not G1.has_edge(mapped_u, mapped_v):
            print(f"Edge ({mapped_u}, {mapped_v}) is missing in G1 (expected from ({u}, {v}) in G2).")
            return False  # A required edge is missing in G1

    # If all nodes and edges match, the mapping is valid
    return True
    

def test_mcs(iterations: int = 100):
    # create common random directed graph
    common_G = nx.gnp_random_graph(6, 0.5, directed=True)
    
    
    for i in range(iterations):
        G1 = common_G.copy()
        G2 = common_G.copy()
        # grow graphs
        G1 = grow_graph(G1, 6, 3)
        G2 = grow_graph(G2, 6, 3)

        test_grow_graph(common_G, G1, G2)
        
        
        # find mcs
        mcs_solver = McSplit(G1, G2, 
                            node_match=node_match_by_type_label,
                            edge_match=edge_match_by_type)
        best_mapping = mcs_solver.find_mcs()
        # Create the common subgraph from the mapping
        common_solution_subgraph = nx.DiGraph()
        for u, v in best_mapping.items():
            # Add nodes with attributes
            common_solution_subgraph.add_node(u, **G1.nodes[u])
            # Add edges that exist in both graphs
            for u2, v2 in best_mapping.items():
                if G1.has_edge(u, u2) and G2.has_edge(v, v2):
                    common_solution_subgraph.add_edge(u, u2, **G1[u][u2])
    
        isomorphic, G_common_size, G_solution_size, com_iter, sol_iter, mapping = (check_solution(G1, G2, common_G, solution=common_solution_subgraph, mapping=best_mapping))
        if isomorphic is False or G_common_size > G_solution_size:
            return False
    return True

def vanila_test():
    # Example Usage
    G1 = nx.DiGraph()
    G1.add_nodes_from([
        (1, {"type": "T", "label": "a"}),
        (2, {"type": "T", "label": "b"}),
        (3, {"type": "T", "label": "c"})
    ])
    G1.add_edges_from([
        (1, 2, {"type": "1"}),
        (2, 3, {"type": "1"})
    ])

    G2 = nx.DiGraph()
    G2.add_nodes_from([
        (10, {"type": "T", "label": "a"}),
        (20, {"type": "T", "label": "b"}),
        (30, {"type": "T", "label": "c"})
    ])
    G2.add_edges_from([
        (10, 20, {"type": "1"}),
        (20, 30, {"type": "1"})
    ])

    # Create McSplit solver with custom matching functions
    mcs_solver = McSplit(G1, G2, 
                        node_match=node_match_by_type_label,
                        edge_match=edge_match_by_type)
    
    # Find maximum common subgraph
    best_mapping = mcs_solver.find_mcs()
    print("\nMaximum Common Subgraph Mapping:", best_mapping)
    
    # Print the size of the common subgraph
    print("Size of common subgraph:", len(best_mapping))
    
    # Create the common subgraph from the mapping
    common_subgraph = nx.DiGraph()
    for u, v in best_mapping.items():
        # Add nodes with attributes
        common_subgraph.add_node(u, **G1.nodes[u])
        # Add edges that exist in both graphs
        for u2, v2 in best_mapping.items():
            if G1.has_edge(u, u2) and G2.has_edge(v, v2):
                common_subgraph.add_edge(u, u2, **G1[u][u2])
    
    print("\nCommon Subgraph:")
    print("Nodes:", common_subgraph.nodes(data=True))
    print("Edges:", common_subgraph.edges(data=True))

def vanila_test_grow_graph():
    # ---------------test grow graph-------------------
    G = nx.gnp_random_graph(4, 0.5, directed=True)
    G2 = G.copy()
    G2 = grow_graph(G2, 6, 8)
    # draw on one image
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    nx.draw(G, with_labels=True)
    plt.subplot(122)
    nx.draw(G2, with_labels=True)
    plt.show()



if __name__ == "__main__":
    vanila_test()
    #vanila_test_grow_graph()
    # test mcs
    print(test_mcs(10))
