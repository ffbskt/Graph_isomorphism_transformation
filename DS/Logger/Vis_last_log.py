import json
import networkx as nx
import matplotlib.pyplot as plt
from DS.Visualisation.visg import VisG


def visualize_last_graph_from_log(log_file):
    """
    Reads a log file and visualizes the last recorded graph transformation.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file containing JSON logs.
    """
    last_graph_pattern = None
    last_replacement = None
    all_replacements = []  # Store all replacements within the last second

    # Read the log file and extract relevant logs
    with open(log_file, 'r') as f:
        for line in f:
            log_entry = json.loads(line)
            if log_entry["message"] == "Graph after add pattern":
                last_graph_pattern = log_entry["graph_pattern"]
            elif log_entry["message"] == "Graph after execute special rules":
                last_replacement = log_entry["graph_pattern"]
                all_replacements.append(last_replacement)

    if not last_graph_pattern or not last_replacement:
        print("❌ No valid graph transformation logs found.")
        return

    # Convert log data into NetworkX graphs
    G_before = create_graph_from_log(last_graph_pattern)
    G_after = create_graph_from_log(last_replacement)

    # Visualize the last transformation
    VisG.visualize_transformation(G_before, G_after, result=G_after, test_name="Last Graph Transformation")

    # Combine all replacements and visualize
    if len(all_replacements) > 1:
        combined_graph = compose_replacement_graphs(all_replacements)
        VisG.visualize_transformation(G_before, combined_graph, result=combined_graph, test_name="Composed Replacements")

def create_graph_from_log(graph_data):
    """
    Creates a NetworkX graph from log JSON structure.
    
    Parameters:
    -----------
    graph_data : dict
        Contains 'Gnodes' and 'Gedges' from the log.
    
    Returns:
    --------
    nx.DiGraph
        Graph constructed from log data.
    """
    G = nx.DiGraph()
    
    # Convert string representation to tuples
    nodes = eval(graph_data["Gnodes"])  # Convert string representation into Python objects
    edges = eval(graph_data["Gedges"])

    # Add nodes
    for node, attr in nodes:
        G.add_node(node, **attr)

    # Add edges
    for src, dst, attr in edges:
        G.add_edge(src, dst, **attr)

    return G

def compose_replacement_graphs(replacement_logs):
    """
    Composes multiple replacement graphs into one final graph.
    
    Parameters:
    -----------
    replacement_logs : list of dict
        List of transformation logs.
    
    Returns:
    --------
    nx.DiGraph
        Combined graph after applying multiple transformations.
    """
    combined_graph = nx.DiGraph()

    for log in replacement_logs:
        G = create_graph_from_log(log)
        combined_graph = nx.compose(combined_graph, G)

    return combined_graph


if __name__ == "__main__":
    log_file = "Log_graph.json"
    visualize_last_graph_from_log(log_file)