import json
import networkx as nx
import matplotlib.pyplot as plt
from DS.Visualisation.visg import VisG
from typing import List, Dict, Optional, Tuple
import os
import argparse


class LogVisualizer:
    def __init__(self, log_file: str):
        """Initialize LogVisualizer with a log file path.

        Parameters:
        -----------
        log_file : str
            Path to the log file containing JSON logs.
        """
        self.log_file = log_file
        self.last_graph_pattern = None
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
        self.last_graph_patterns = []
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            
            # If no specific ids provided, get the last 5 entries
            if ids is None:
                ids = range(len(lines)-5, len(lines))    
                # Get entries with specific IDs
            for line in reversed(lines):
                log_entry = json.loads(line)
                if 'id' in log_entry and log_entry["message"] in graphs_type and log_entry["id"] in ids:
                    self.last_graph_patterns.append((log_entry["id"], log_entry["message"], log_entry["graph_pattern"]))
                if len(self.last_graph_patterns) == last_n:
                    break
        return reversed(self.last_graph_patterns)

    def create_graphs_from_log(self, last_graph_patterns: List[Tuple[int, str, Dict]]) -> List[Tuple[str, nx.DiGraph]]:
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
        for log_entry in last_graph_patterns:
            # Unpack the tuple (id, message, graph_pattern)
            id_, message, graph_pattern = log_entry
            graph = self._create_graph_from_log(graph_pattern)
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
    parser = argparse.ArgumentParser(description='Visualize graph logs')
    parser.add_argument('--last_n', type=int, default=5, help='Number of last entries to retrieve')
    parser.add_argument('--id_start', type=int, help='Start of ID range')
    parser.add_argument('--id_end', type=int, help='End of ID range (inclusive)')
    args = parser.parse_args()

    log_file = "Log_graph.json"
    visualizer = LogVisualizer(log_file)
    
    # Set up ids range if provided
    ids = None
    if args.id_start is not None and args.id_end is not None:
        ids = range(args.id_start, args.id_end + 1)
    
    # Get last logs
    last_logs = visualizer.get_last_n_logs(ids=ids, last_n=args.last_n)
    
    # Create graphs from logs
    graphs = visualizer.create_graphs_from_log(last_logs)
    
    # Visualize the sequence of graphs
    visual_plot = VisG()
    visual_plot.visualize_graph_from_logs(graphs)
