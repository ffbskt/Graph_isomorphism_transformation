import sys
import os
import random
import datetime
# sys.path.append(r'C:\Users\Denis\PycharmProjects\Graph_isomorphism_transformation\visg')
from Visualisation.visg import VisG
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

random.seed(1)


def create_random_graph(num_nodes, num_edges, node_types=None, edge_types=None, node_labels=None, edge_labels=None):
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
        node_types = ['a', 'b', 'c']
    if node_labels is None:
        node_labels = ['a', 'b', 'c']
    if edge_types is None:
        edge_types = [1, 2, 3]
    if node_labels is None:
        node_labels = ['a', 'b', 'c']

    # Create a random graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        node_type = random.choice(node_types)
        node_label = random.choice(node_labels)
        G.add_node(i, type=node_type, label=node_label)

        # Add edges
    for i in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        edge_type = random.choice(edge_types)
        edge_label = random.choice(edge_labels)
        G.add_edge(src, dst, type=edge_type, label=edge_label)

    return G


def create_pattern(phead, pbase, edges):
    """
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
    return P


def subgraph_with_neighbors(G, node_list, depth=1, only_out_edges=False):
    nodes_to_add = set(node_list)

    for node in node_list:
        queue = deque([(node, 0)])

        while queue:
            current_node, current_depth = queue.popleft()

            if current_depth < depth:
                neighbors = list(G.neighbors(current_node))

                if not only_out_edges:
                    # Also considering parents as well as children by looking at predecessor nodes
                    predecessors = list(G.predecessors(current_node))
                    neighbors.extend(predecessors)

                for neighbor in neighbors:
                    if neighbor not in nodes_to_add:
                        nodes_to_add.add(neighbor)
                        queue.append((neighbor, current_depth + 1))

    if only_out_edges:
        nodes_to_add = nodes_to_add - set(node_list)

    return G.subgraph(nodes_to_add)


def find_isomorphisms(pattern_base, G, node_match=None, edge_match=None):
    return nx.algorithms.isomorphism.GraphMatcher(
        G, pattern_base, node_match=node_match, edge_match=edge_match).subgraph_isomorphisms_iter()


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
        # print('replace', replace, G.nodes(data=True), G.edges(data=True))
    pbase = subgraph_with_neighbors(G, ['B', ], depth=1)
    nodes_to_remove = list(pbase.nodes())
    # print('nodes_to_remove', nodes_to_remove)
    G.remove_nodes_from(nodes_to_remove)

    # print('G', G.edges(data=True), G.nodes(data=True))


def apply_pattern(pattern, G):
    """
    Apply pattern to G
    """
    for iso in find_isomorphisms(pattern, G):
        replace_by_isomorphism(pattern, G, iso)


def visualize_transformation(graph, pattern, result, test_name):
    """
    Visualize the graph transformation process with three subplots and save to file.
    All visualizations will be saved in a single file, arranged vertically.

    Args:
        graph (nx.DiGraph): Initial graph
        pattern (nx.DiGraph): Pattern graph
        result (nx.DiGraph): Resulting graph after transformation
        test_name (str): Name of the test for the title
    """
    # Create figure for this test
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Initial graph
    vis1 = VisG()
    vis1.add_graph(graph)
    vis1.draw(layout='spring', title='Initial Graph', ax=ax1)

    # Pattern graph
    vis2 = VisG()
    vis2.add_graph(pattern)
    vis2.draw(layout='spring', title='Pattern', ax=ax2)

    # Result graph
    vis3 = VisG()
    vis3.add_graph(result)
    vis3.draw(layout='spring', title='Result Graph', ax=ax3)

    plt.suptitle(test_name)
    plt.tight_layout()

    # Save the visualization to file
    output_dir = "test_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a filename for the combined image
    combined_path = os.path.join(output_dir, "combined_transformations.png")
    
    # If this is the first visualization, create new file
    # If not, append to existing file
    if not hasattr(visualize_transformation, 'figures'):
        visualize_transformation.figures = []
    
    # Store the current figure
    visualize_transformation.figures.append((fig, test_name))
    
    # If this is the last test (checking if we're in the main test sequence)
    if "Test 4:" in test_name or "Test 5:" in test_name:
        # Create a new figure for all tests combined
        n_tests = len(visualize_transformation.figures)
        combined_fig = plt.figure(figsize=(18, 5 * n_tests))
        
        # Add each test visualization as a subplot
        for i, (test_fig, title) in enumerate(visualize_transformation.figures):
            # Get the test figure canvas
            canvas = test_fig.canvas
            canvas.draw()
            
            # Create new subplot in the combined figure
            ax = combined_fig.add_subplot(n_tests, 1, i + 1)
            
            # Remove axes
            ax.axis('off')
            
            # Add the test figure as an image
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))
            ax.imshow(img)
            ax.set_title(title)
            
            # Close the individual test figure
            plt.close(test_fig)
        
        # Save the combined figure
        combined_fig.tight_layout()
        combined_fig.savefig(combined_path, bbox_inches='tight', dpi=300)
        plt.close(combined_fig)
        
        # Clear the stored figures
        visualize_transformation.figures = []
        
        print(f"Saved combined visualization to {combined_path}")
    else:
        print(f"Added {test_name} to combined visualization")

# def visualize_transformation(graph, pattern, result, test_name):
#     """Alias for visualize_transformation2 for backward compatibility."""
#     return visualize_transformation2(graph, pattern, result, test_name)


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
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'b'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 'replacement', 'label': '1'})])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
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
        visualize_transformation(graph2_copy, pattern, graph2, "Test 1: Single Node Replacement")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_multiple_node_replacement(graph2=create_test_graph(type='linear'), visualize=False):
    """Test replacing a node with multiple connected nodes."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
        (12, {'type': 'F', 'label': 'b'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 'replacement', 'label': '1'}),
        (10, 12, {'type': 'replacement', 'label': '1'})
    ])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
        (12, {'type': 'F', 'label': 'b'}),
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
        visualize_transformation(graph2_copy, pattern, graph2, "Test 2: Multiple Node Replacement")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding a new node to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([(11, {'type': 'F', 'label': 'b'})])
    pattern = create_pattern(phead, pbase, [(10, 11, {'type': 1, 'label': '1'})])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
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
        visualize_transformation(graph2_copy, pattern, graph2, "Test 3: Node Addition")

    return nx.utils.graphs_equal(result_graph, graph2)


def test_multiple_node_addition(graph2=create_test_graph(type='linear'), visualize=False):
    """Test adding multiple connected nodes to existing node."""
    # Create pattern
    pbase, phead = nx.DiGraph(), nx.DiGraph()
    pbase.add_nodes_from([(10, {'type': 'T', 'label': 'a'})])
    phead.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
        (12, {'type': 'F', 'label': 'b'}),
    ])
    phead.add_edges_from([(11, 12, {'type': 1, 'label': '1'})])
    pattern = create_pattern(phead, pbase, [
        (10, 11, {'type': 1, 'label': '1'}),
        (10, 12, {'type': 1, 'label': '1'})
    ])

    # Create expected result
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from([
        (11, {'type': 'F', 'label': 'b'}),
        (12, {'type': 'F', 'label': 'b'}),
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
        visualize_transformation(graph2_copy, pattern, graph2, "Test 4: Multiple Node Addition")

    return nx.utils.graphs_equal(result_graph, graph2)


if __name__ == "__main__":
    # Run all tests
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

    G = nx.DiGraph()
    G.add_nodes_from([(0, {'type': 'a', 'label': 'a'}),
                      (1, {'type': 'b', 'label': 'b'}),
                      (2, {'type': 'a', 'label': 'a'}),
                      ])
    G.add_edges_from([(0, 1, {'type': 1, 'label': '1'}),
                      (2, 0, {'type': 1, 'label': '1'}),
                      (1, 2, {'type': 1, 'label': '1'}),
                      ])

    phead, pbase = nx.DiGraph(), nx.DiGraph()
    phead.add_nodes_from([(10, {'type': 'T', 'label': 'a'}),
                          (11, {'type': 'F', 'label': 'b'}),
                          ])
    phead.add_edges_from([(10, 11, {'type': 1, 'label': '1'}),
                          ])
    # pattern = create_pattern(phead, pbase, [(10, 11, {'type': 1, 'label': '1'}),])
    # apply_pattern(pattern, G)
    # visualize_transformation(G, pattern, G, "Test 5: Multiple Node Addition")




