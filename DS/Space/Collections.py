import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from DS.Space.Matching import edge_none_match, node_none_match
from DS.Logger.logger import JSONLogger
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

COMAND_EDGE_TYPES = {
    'replacement': 'r',
    'replacement_in': 'ri',
    'replacement_out': 'ro'
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
    def __init__(self, NC=None, logger=None):
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
        self.logger = logger
        if logger is None:
            self.logger = JSONLogger()

    def clear(self):
        self.NC = NodeCollection()
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
        graph_log = self.get_graph_to_collection_log(G_reindexed, reindex_map)
        self.logger.info("Graph added to collection p={}".format(is_pattern), graph_pattern=graph_log)
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

    # __________________________Transformations_________________________

    def renew_iso(self, iso, reindex_map):
        new_iso = {}
        for old, new in iso.items():
            new_iso[old] = reindex_map[new]
        return new_iso

    def add_copy_of_pattern_to_G(self, pattern):
        P_copy_as_graph = pattern.copy()
        # remove B, H nodes from pattern
        P_copy_as_graph.remove_nodes_from(['B', 'H'])
        _, reindex_map = self.add_graph_to_collection(P_copy_as_graph, label=None, is_pattern=False)
        return P_copy_as_graph, reindex_map
        
        
    def transfer_data(self, src_node, dst_node, head=True):
        if head:
            if dst_node['type'] == '^X':
                dst_node['type'] = src_node['type']
            if dst_node['label'] == '^X':
                dst_node['label'] = src_node['label']
        else: # mean base
            if dst_node['type'] == None:
                dst_node['type'] = src_node['type']
            if dst_node['label'] == None:
                dst_node['label'] = src_node['label']

    def update_pairs(self, pairs, base_ind, head_ind):
        updated_pairs = []
        for pair in pairs:
            updated_pair = tuple(head_ind if value == base_ind else value for value in pair)
            updated_pairs.append(updated_pair)
        return updated_pairs
            

    def execute_spetial_rules(self, pattern, reindex_map):
        """
        replacement - take data from phead node, remove pbase, save edges.
        """
        nx.relabel_nodes(pattern, reindex_map, copy=False)
        nx.relabel_nodes(pattern.graph['pbase'], reindex_map, copy=False)
        edges_to_replace = defaultdict(list)
        nodes_to_remove = set()
        
        # First, find all replacement edges and mark nodes for removal
        for u, v, d in list(pattern.edges(data=True)):
            if d['type'] == 'replacement':
                # Remove the replacement edge from the pattern so it doesn't get added to G
                
                nodes_to_remove.add(u)
                self.transfer_data(self.G.nodes[u], self.G.nodes[v], head=True)
                
                # Store all incoming edges to base node
                for pred in list(self.G.predecessors(u)):
                    edge_data = self.G.get_edge_data(pred, u)
                    if edge_data['type'] not in COMAND_EDGE_TYPES:
                        edges_to_replace[(u, v)].append(('in', pred, edge_data))
                    
                
                # Store all outgoing edges from base node
                for succ in list(self.G.successors(u)):
                    edge_data = self.G.get_edge_data(u, succ)
                    if edge_data['type'] not in COMAND_EDGE_TYPES:
                        edges_to_replace[(u, v)].append(('out', succ, edge_data))
                    
         # Process each replacement pair
        for (base_node, head_node), edges_list in edges_to_replace.items():
            # Add new edges
            for direction, other_node, edge_data in edges_list:
                if direction == 'in':
                    # Add edge from predecessor to head node
                    if other_node != head_node:  # Prevent self-loops
                        self.G.add_edge(other_node, head_node, **edge_data)
                else:  # direction == 'out'
                    # Add edge from head node to successor
                    if other_node != head_node:  # Prevent self-loops
                        self.G.add_edge(head_node, other_node, **edge_data)
        
        # Remove base nodes that were replaced (this will also remove all their edges)
        for node in nodes_to_remove:
            if node in self.G:
                self.G.remove_node(node)

    def transform(self, pattern, number_of_transformations=1, visualize=False):
        """
        Apply the transformation pattern to graph G
        
        Args:
            G (nx.DiGraph): Graph to transform
            visualize (bool): Whether to visualize the transformation
            
        Returns:
            nx.DiGraph: Transformed graph
        """
        pbase = pattern.graph['pbase']
        isomorphisms = nx.algorithms.isomorphism.DiGraphMatcher(self.G, pbase, node_match=node_none_match, edge_match=edge_none_match).subgraph_isomorphisms_iter()
        isomorphisms = list(isomorphisms)[:number_of_transformations]
        self.logger.info("Isomorphisms found", isomorphisms=isomorphisms, pattern=pattern.nodes(data=True))
        for iso in isomorphisms:
            pattern_copy, reindex_map = self.add_copy_of_pattern_to_G(pattern)
            iso = self.renew_iso(iso, reindex_map)
            #print('-------new iso', iso, reindex_map,  pattern_copy.edges(), self.G.edges())
            nx.relabel_nodes(self.G, iso, copy=False)
            #print('-------new graph', self.G.edges())
            log = self.get_transformation_log(pattern, iso, depth=None)
            self.logger.info("Graph after add pattern", graph_pattern=log) # add base instead of G onodes 
            self.execute_spetial_rules(pattern_copy, reindex_map)
            self.logger.info("Graph after execute special rules", graph_pattern=log) # add base instead of G onodes 

    def get_transformation_log(self, pattern, iso, depth=2):
        active_nodes = []
        for k, v in iso.items():
            active_nodes.extend([k, v])
        if depth is None:
            g = self.G
        else:
            g = self.subgraph_with_neighbors(node_list=active_nodes, depth=depth)
        return {"Gnodes": g.nodes(data=True), "Gedges": g.edges(data=True), 
                "Pbase": pattern.graph['pbase'].nodes(), "Phead": pattern.graph['phead'].nodes()}

    def get_graph_to_collection_log(self, graph, reindex_map):
        return {"Gnodes": graph.nodes(data=True), "Gedges": graph.edges(data=True), 
                "reindex_map": reindex_map}
            



if __name__ == "__main__":
    pass