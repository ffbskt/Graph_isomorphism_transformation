import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import copy
import random
random.seed(1)

# Constants for graph types and labels
SINGLETON_TYPES = {
    'HIERARCHY': 'H',
    'TIME': 'T',
    'LABEL': 'L',
    'PATTERN': 'P',   
}

EDGE_TYPES = {
    'LABEL': 'l', # Just for add hierarchy node to Label base node.
    'HIERARCHY': 'h', 
    'TIME': 't',
    #'LABEL HORIZONTAL': 'lh', # ???
    'HIERARCHY HORIZONTAL': 'hh',
    'TRANSFORMATION HORIZONTAL': 'th'
}



class NodeCollection:
    """Manages unique node identifiers and their properties across all graphs.
    
    This class serves as a global registry for nodes, ensuring unique IDs and 
    maintaining mappings between node properties and their instances.
    """
    def __init__(self):
        self.nodes_collection = {}
        self.type2nodes = defaultdict(list)
        self.label2nodes = defaultdict(list)
        
    def __getitem__(self, key):
        return self.nodes_collection[key]

    def __contains__(self, key):
        return key in self.nodes_collection

    def __setitem__(self, key, value):
        self.nodes_collection[key] = value
        self.type2nodes[value['type']].append(key)
        self.label2nodes[value['label']].append(key)

    def __len__(self):
        return len(self.nodes_collection)

    def add_new_node(self, node_data):
        """Creates a new node with unique ID and registers its properties.
        
        Args:
            node_data (dict): Node properties including 'type' and 'label'
            
        Returns:
            int: Unique node identifier
        """
        node_id = len(self.nodes_collection)
        self.nodes_collection[node_id] = node_data
        
        # Register node properties for quick lookup
        # print(node_data, self.label2nodes)
        for data_name in ['type', 'label']:
            if data_name in node_data:
                getattr(self, data_name + '2nodes')[node_data[data_name]].append(node_id)
        return node_id

    # find all nodes with same data as in pattern base
    def find_nodes_with_same_data_in_loc(self, nodes_data):
        # node_data_dict = [{type: 'ch', sign: 'pos', intensity: 0.5, coordinate: (0, 0)}..] or graph.nodes(data=True)
        all_matched_nodes = set()
        # takde node from pattern base and find all nodes with same data in loc
        for ind_node_data in nodes_data:
            one_node_matched_nodes = set()
            _, node_data = ind_node_data
            if is_None_for_any(node_data):
                # TODO: how work with None in one or two variables?
                # TODO: find all nodes with None ...
                continue
            for i, data_name in enumerate(node_data):
                nodes = getattr(self, data_name + '2nodes')[node_data[data_name]]
                # if one_node_matched_nodes is empty, then add all nodes else find intersection
                if not nodes:
                    one_node_matched_nodes = set()
                    break
                elif i == 0:
                    one_node_matched_nodes = set(nodes)
                else:
                    one_node_matched_nodes = one_node_matched_nodes.intersection(set(nodes))
            all_matched_nodes = all_matched_nodes.union(one_node_matched_nodes)
        return all_matched_nodes
    

    # def get_subgraph_by_pattern(self, G, pattern):
    #     all_matched_nodes = self.find_nodes_with_same_data_in_loc(pattern.base_graph.nodes(data=True))
    #     sub_graphs = self.subgraph_with_neighbors(G, all_matched_nodes, depth=pattern.base_size_not_none)
    #     return sub_graphs
    

class GraphCollection:
    """Manages a hierarchical collection of graphs with transformation capabilities.
    
    This class maintains a hierarchy of graphs, their relationships, and supports
    graph transformations through patterns. It ensures proper labeling and maintains
    relationships between different graph components.
    
    Attributes:
        G (nx.DiGraph): Main graph storing all nodes and their relationships
        label2graphs (dict): Maps labels to graph identifiers
        singleton2id (dict): Maps singleton node types to their unique IDs
    """
    def __init__(self, NC=None):
        """
        [
        Notes:
        MC - morphism composition (triangle relation show property or relation dimention)
        Hierarhical graph collection.
        Important that we maintance:
        Label: Graph, because same nodes could have different links
         It is mean one difinition have different implementation. Then we should gather its in MC.
         Also it give us to use label without implementation also by MC. 
        Hierarchy: Going close to Label and put each new label in hierarchy.
        Time: show current state scrin to distinct of prehistory.
        Or the same graphs (eq: nodes, links) could be different in time or other struncture upper hierarchy. 
        (Graph nodes could be different on next next. But always equal in down - pred pred, but ?? next pred I think now)
        Summarize: Rule - hierarchy two equal leaves have always same predcessor label, but diff time.

        Maintaine Functions:
        hierarchy_maintaine, time_maintaine, label_maintaine, parent_maintaine
        Pattern is spetial structure that unique for time. (No two equal pattern in different time)
        
        ] 
        """
        if NC is None:
            NC = NodeCollection()
        self.NC = NC
        self.G = nx.DiGraph()
        self.label2graphs = {}
        self.label2hierarchy = defaultdict(list) # one label could be word or set of words  # Maps hierarchy level to node ID
           
    def add_node(self, node_data):
        indx = self.NC.add_new_node(node_data)
        self.G.add_node(indx, **node_data)
        return indx
    
    def add_label_to_pattern(self, P, label=None):
        # rebuild this two methods to one pattern/graph true false, add all nodes before Hyrarchy typw edge.
        assert False, "Debricated method"
        if label is None:
            label = f"{len(self.G)},."
        assert label not in self.label2graphs, f"Label '{label}' already exists"

        graph_id = self.NC.add_new_node({'type': 'P', 'label': label})
        self.G.add_node(graph_id, type='P', label=label)
        self.label2graphs[label] = graph_id

        for node, data in P.nodes(data=True):
            
            if data['type'] in ['B', 'H']:
                self.G.add_edge(graph_id, node, type=EDGE_TYPES['HIERARCHY'], label='')



    def add_label(self, G, label=None): # add type G/P
        """Maintains graph labeling and hierarchy relationships. == create new variable.
        
        Args:
            G (nx.DiGraph): Graph to be labeled
            label (str, optional): Custom label for the graph
            
        Raises:
            AssertionError: If label is not unique
        """
        if label is None:
            label = f"{len(self.G)},."
            
        assert label not in self.label2graphs, f"Label '{label}' already exists"
        
        # Create root graph node
        graph_id = self.NC.add_new_node({'type': 'G', 'label': label})
        self.G.add_node(graph_id, type='G', label=label)
        self.label2graphs[label] = graph_id
        
        # Connect graph node to its components
        for node in G.nodes():
            self.G.add_edge(graph_id, node, type=EDGE_TYPES['HIERARCHY'], label='')
            
        # Connect to label hierarchy
        assert len(self.NC.label2nodes[SINGLETON_TYPES['LABEL']]) == 1, f"Label '{label}' has more than one node or not initializ by init graph"
        # print(self.NC.label2nodes, SINGLETON_TYPES['LABEL'], self.NC.label2nodes[SINGLETON_TYPES['LABEL']])
        self.G.add_edge(
            self.NC.label2nodes[SINGLETON_TYPES['LABEL']][0], 
            graph_id, 
            type=EDGE_TYPES['LABEL'], 
            label='l'
        )

    def reindex_pattern(self, P, reindex_map):
        P.graph['pbase'] = nx.relabel_nodes(P.graph['pbase'], reindex_map)
        P.graph['phead'] = nx.relabel_nodes(P.graph['phead'], reindex_map)
        P.graph['iB'] = reindex_map[P.graph['iB']]
        P.graph['iH'] = reindex_map[P.graph['iH']]
        return P
    
    def add_graph_to_collection(self, G, label=None, is_pattern=False):
        """Adds a graph to the graph collection as new subgraph.
        
        Args:
            G (nx.DiGraph): Graph to add
            ?? not needed, add as a simople graph is_pattern (bool): Whether the graph is a pattern
            
        Returns:
            nx.DiGraph: New graph with unique node IDs
        """
        reindex_map = {old: self.NC.add_new_node(G.nodes[old]) for old in G.nodes()}
        G_reindexed = nx.relabel_nodes(G, reindex_map)
        self.G.update(G_reindexed)
        #self.add_label(G_reindexed, label)
        if is_pattern:
            G_reindexed = self.reindex_pattern(G_reindexed, reindex_map)
        return G_reindexed, reindex_map
    
    def subgraph_with_neighbors(self, node_list, G=None, depth=1, only_out_edges=False, remove_nodes=[]):
        """Get subgraph containing nodes and their neighbors up to specified depth.
        
        Args:
            node_list (list): List of nodes to start from
            G (nx.DiGraph, optional): Graph to search in. Defaults to self.G
            depth (int, optional): How many layers of neighbors to include. Defaults to 1
            only_out_edges (bool, optional): Whether to only follow outgoing edges. Defaults to False
            remove_nodes (list, optional): Nodes to exclude from result. Defaults to []
            
        Returns:
            nx.DiGraph: Subgraph containing specified nodes and their neighbors
        """
        if G is None:
            G = self.G
            
        nodes_to_add = set()
        
        # Add initial nodes
        for node in node_list:
            if node in G:
                nodes_to_add.add(node)
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
        nodes_to_add = nodes_to_add - set(remove_nodes) # remove nodes from nodes_to_add mostly do not include start of search

        return G.subgraph(nodes_to_add)

    
   # This wrong method, isomorphism instead is correct. (iso base: node-^X type...)
    # def get_concrete_edge(self, node, G=None, elabel=None, etype=None, for_children=True):
    #     if G is None:
    #         G = self.G
    #     if for_children:
    #         childrens = list(G.successors(node))
    #     else:
    #         childrens = list(G.predecessors(node))
    #     for child in childrens:
    #         if for_children:
    #             edge = (node, child)
    #         else:
    #             edge = (child, node)
    #         if self.edge_param_match(edge, G, elabel, etype):
    #             return edge
    #     return None
    
