import json
import networkx as nx
import matplotlib.pyplot as plt
from DS.Visualisation.visg import VisG
from typing import List, Dict, Optional, Tuple
import os
import argparse


class LogReader:
    def __init__(self, log_file: str):
        """Initialize LogReader with a log file path.

        Parameters:
        -----------
        log_file : str
            Path to the log file containing JSON logs.
        """
        self.log_file = log_file
        self.last_graph = None
        self.last_replacement = None
        self.all_replacements = []

    def get_last_n_logs(self, graphs_type: list=None, ids: list=None, last_n: int=5) -> List[Tuple[int, str, Dict]]:
        """Get the graph transformations from the log.

        Parameters:
        -----------
        graphs_type : list, optional
            List of graph types to retrieve. If None, gets the last 5 entries.  
        ids : list, optional
            List of specific log entry IDs to retrieve. If None, gets the last 5 entries.
        last_n : int, optional
            Number of last entries to retrieve. Default is 5.

        Returns:
        --------
        List[Tuple[int, str, Dict]]
            List of tuples containing (id, message, graph_pattern) for each log entry
        """
        if graphs_type is None:
            graphs_type = ["Graph after add pattern", "Graph after execute special rules", 
                           "Graph added to collection", "Graph added to collection p=False", 
                           "Graph added to collection p=True"]        
        self.last_graphs = []
        print('---', last_n, ids, graphs_type)
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            
            # If no specific ids provided, get the last 5 entries
            if ids is None:
                ids = range(len(lines)-5, len(lines))    
                # Get entries with specific IDs
            for line in reversed(lines):
                log_entry = json.loads(line)
                if 'id' in log_entry and log_entry["message"] in graphs_type and log_entry["id"] in ids:
                    self.last_graphs.append((log_entry["id"], log_entry["message"], log_entry["graph"]))
                if len(self.last_graphs) == last_n:
                    break
        return reversed(self.last_graphs)

    def create_graphs_from_log(self, last_graphs: List[Tuple[int, str, Dict]]) -> List[Tuple[str, nx.DiGraph]]:
        """Create NetworkX graphs from log entries.

        Parameters:
        -----------
        last_graph_patterns : List[Tuple[int, str, Dict]]
            List of tuples containing (id, message, graph_pattern)

        Returns:
        --------
        List[Tuple[str, nx.DiGraph]]
            List of tuples containing graph name and NetworkX graph object
        """
        graphs_from_log = []
        for log_entry in last_graphs:
            # Unpack the tuple (id, message, graph)
            id_, message, graph = log_entry
            graph = self._create_graph_from_log(graph)
            name = f"{id_}_{message}"
            graphs_from_log.append((name, graph))
        return graphs_from_log


    @staticmethod
    def _create_graph_from_log(graph_data: Dict) -> nx.DiGraph:
        """Creates a NetworkX graph from log JSON structure.

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize graph patterns from log file')
    parser.add_argument('--log_file', type=str, default="Log_graph.json", help='Path to the log file')
    parser.add_argument('--last_n', type=int, default=5, help='Number of last entries to retrieve')
    parser.add_argument('--id_start', type=int, help='Start of ID range')
    parser.add_argument('--id_end', type=int, help='End of ID range (inclusive)')
    parser.add_argument('--graphs_type', nargs='+', choices=[
        'Pattern_graph',
        'Source_graph',
        'Target_graph',
        "Graph_after_add_pattern", "Graph_after_execute_special_rules", 
        "Graph_added_to_collection", "Graph_added_to_collection_p=False","Graph_added_to_collection_p=True"
    ], default=['Graph_after_add_pattern', 'Graph_after_execute_special_rules', 'Graph_added_to_collection', 'Graph_added_to_collection_p=False', 'Graph_added_to_collection_p=True'], 
    help='Types of graphs to retrieve. Can specify multiple types.')

    args = parser.parse_args()

    # Convert graph type choices to actual message strings
    type_mapping = {
        'Pattern_graph': 'Pattern graph',
        'Source_graph': 'Source graph',
        'Target_graph': 'Target graph',
        'Graph_after_add_pattern': 'Graph after add pattern',
        'Graph_after_execute_special_rules': 'Graph after execute special rules',
        'Graph_added_to_collection': 'Graph added to collection',
        'Graph_added_to_collection_p=False': 'Graph added to collection p=False',
        'Graph_added_to_collection_p=True': 'Graph added to collection p=True'
    }
    graphs_type = [type_mapping[t] for t in args.graphs_type]

    log_file = args.log_file
    last_n = args.last_n
    id_start = args.id_start
    id_end = args.id_end

    # Create reader and get graphs
    reader = LogReader(log_file)
    
    # Set up ids range if provided
    ids = None
    if args.id_start is not None and args.id_end is not None:
        ids = range(args.id_start, args.id_end + 1)
    
    # Get last logs
    last_logs = reader.get_last_n_logs(ids=ids, last_n=args.last_n, graphs_type=graphs_type)
    
    # Create graphs from logs
    graphs = reader.create_graphs_from_log(last_logs)
    
    # Visualize the sequence of graphs
    visual_plot = VisG()
    visual_plot.visualize_graph_from_logs(graphs)
