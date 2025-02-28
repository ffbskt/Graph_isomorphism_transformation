from collections import defaultdict
import networkx as nx
from collections import deque

from copy import deepcopy

def is_None_for_any(node_data):
    for name in node_data:
        if node_data[name] is None:
            return True
    return False


def convert_dict(dict):
    return {value: key for key, value in dict.items()}

def node_match(node, pattern_node_data):
    #print(123)
    for key in pattern_node_data:
        #print(pattern, pattern[key] != node.get(key))
        if pattern_node_data[key] is not None and pattern_node_data[key] != node.get(key):
            return False
    return True

def add_node(G, node_collection, node_data=None):
    index = node_collection.add_new_node(node_data)
    G.add_node(index, **node_data)
    return index

def debug_print(G, *args, **kwargs):
    print(*args, **kwargs, end=': ')
    for edge in G.edges():
        if 'sign' in G.nodes[edge[0]]:
            print(edge[0], G.nodes[edge[0]]['sign'],'->', end='')
        else:
            print(edge[0], 'no sign', '->',end='')
    if 'sign' in G.nodes[edge[1]]:
        print(edge[1], G.nodes[edge[1]]['sign'])
    else:
        print(edge[1], 'no sign', '->',end='')

class Pattern:
    def __init__(self, base, head, data):
        self.base_graph = self.create_graph(base, data)
        self.head_graph = self.create_graph(head, data)
        self.base_size_not_none = self.find_not_none_size(self.base_graph)
        self.initial_graphs = deepcopy([self.base_graph, self.head_graph])

    def find_not_none_size(self, graph):
        size = 0
        for node_id, node_data  in graph.nodes(data=True):
            if not is_None_for_any(node_data):
                size += 1
        return size

    #@staticmethod
    def create_graph(self, nodes, data):
        graph = nx.DiGraph()
        for node in nodes:
            if len(node) == 2:
                graph.add_node(node[0], **data.get(node[0], {}))
                graph.add_node(node[1], **data.get(node[1], {}))
                graph.add_edge(*node)
            elif len(node) == 1:
                graph.add_node(node[0], **data.get(node[0], {}))
        return graph

    def clean(self):
        self.__revert_to_initial(self.base_graph, self.initial_graphs[0])
        self.__revert_to_initial(self.head_graph, self.initial_graphs[1])

    def __revert_to_initial(self, current_graph, initial_graph):
        # TODO stupid method and maybe create full transfer function or method here. From G1->G2
        for node, data in current_graph.nodes(data=True):
            initial_data = initial_graph.nodes[node]
            for attr in list(data.keys()):  # Make a copy of keys since we'll modify the dict
                if attr not in initial_data:
                    del current_graph.nodes[node][attr]
                else:
                    current_graph.nodes[node][attr] = initial_graph.nodes[node][attr]

        for edge in current_graph.edges:
            if edge not in initial_graph.edges:
                current_graph.remove_edge(edge)

class NodeCollection(object):
    # Each node in graph has unique index.
    # There all data collection to fast filtering nodes by data and find isomorphisms on subgraph.
    def __init__(self):
        self.nodes_collection = {}
        # need collection for all node data: sign, type, coordinate, intensity
        self.type2nodes = defaultdict(list)
        self.sign2nodes = defaultdict(list)
        self.intensity2nodes = defaultdict(list)
        self.coordinate2nodes = defaultdict(list)
        self.word2nodes = defaultdict(list)
        # TODO: need to add intervals for each graph {self: [0, 127], ...}
        self.graph2nodes_intervals = defaultdict(list)  # intervals of nodes numbers for each graph

    def __getitem__(self, key):
        # if key in self.nodes_collection:
        return self.nodes_collection[key]

    def __contains__(self, key):
        return key in self.nodes_collection


    def __setitem__(self, key, value):
        self.nodes_collection[key] = value

    def __len__(self):
        return len(self.nodes_collection)

    def add_new_node(self, node_data):
        node_id = len(self.nodes_collection)
        self.nodes_collection[node_id] = node_data
        for data_name in ['type', 'sign', 'word', 'intensity', 'coordinate']:
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

    def subgraph_with_neighbors(self, G, node_list, depth=1):
        nodes_to_add = set(node_list)
        for node in node_list:
            queue = deque([(node, 0)])
            while queue:
                current_node, current_depth = queue.popleft()
                if current_depth < depth:
                    neighbors = list(G.neighbors(current_node))
                    # Also considering parents as well as children by looking at predecessor nodes
                    predecessors = list(G.predecessors(current_node))
                    for neighbor in neighbors + predecessors:
                        if neighbor not in nodes_to_add:
                            nodes_to_add.add(neighbor)
                            queue.append((neighbor, current_depth + 1))
        return G.subgraph(nodes_to_add)

    def get_subgraph_by_pattern(self, G, pattern):
        all_matched_nodes = self.find_nodes_with_same_data_in_loc(pattern.base_graph.nodes(data=True))
        sub_graphs = self.subgraph_with_neighbors(G, all_matched_nodes, depth=pattern.base_size_not_none)
        return sub_graphs


def find_isomorphisms(G, pattern_base, node_match=None, edge_match=None):
    return nx.algorithms.isomorphism.GraphMatcher(
        G, pattern_base, node_match=node_match, edge_match=edge_match).subgraph_isomorphisms_iter()

def find_isomorphisms_with_loc(G, pattern_base, node_collection, node_match=None):
    node_list = node_collection.find_nodes_with_same_data_in_loc(pattern_base.nodes(data=True))
    subG = node_collection.subgraph_with_neighbors(G, node_list)
    return find_isomorphisms(subG, pattern_base, node_match=node_match)



# --------------------------Graph Transformations------------------------------------------------------------

def transform_remove_nodes(conv_iso, pattern, G_base):
    for base_node in pattern.base_graph.nodes():
        if base_node not in pattern.head_graph.nodes():
            if conv_iso[base_node] in G_base.nodes():
                G_base.remove_node(conv_iso[base_node])
            else:
                print('remove node no node in base', base_node, conv_iso[base_node])


def transform_remove_edges(conv_iso, pattern, G_base):
    for base_edge in pattern.base_graph.edges():
        if base_edge not in pattern.head_graph.edges():
            #print('remove edge', base_edge, pattern.head_graph.edges())
            print(123, conv_iso[base_edge[0]], conv_iso[base_edge[1]], G_base.edges())
            if (conv_iso[base_edge[0]], conv_iso[base_edge[1]]) in G_base.edges():
                G_base.remove_edge(conv_iso[base_edge[0]], conv_iso[base_edge[1]])
            else:
                print('remove edge no edge in base', base_edge, conv_iso[base_edge[0]], conv_iso[base_edge[1]])


def transform_add_nodes_on(conv_iso, pattern, G_base, node_collection):
    # Add nodes from pattern head to G_out
    iso_head2out = {}
    for node in pattern.head_graph.nodes():
        if node not in conv_iso:
            index = add_node(G_base, node_collection, pattern.head_graph.nodes[node])
            iso_head2out[node] = index
    iso_head2out.update(conv_iso)
    return iso_head2out


def transform_add_edges_on(conv_iso, pattern, G_base):
    for edge in pattern.head_graph.edges():
        if edge not in G_base.edges():
            G_base.add_edge(conv_iso[edge[0]], conv_iso[edge[1]])


def transform_G_base_on(iso, pattern, G_base, node_collection):
    iso_head2base = transform_add_nodes_on(iso, pattern, G_base, node_collection)
    transform_add_edges_on(iso_head2base, pattern, G_base)
    transform_remove_nodes(iso_head2base, pattern, G_base)
    transform_remove_edges(iso_head2base, pattern, G_base)
    return G_base

#-------------------------------------------------------------------------------------------

def get_isomorphisms_on(G_base, pattern, loc=True, node_collection=None, node_match=None, edge_match=None):
    # Instead of get_isomorphisms_from if nodes already exist leave it or remove.
    # Edit nodes in G_base match with pattern nodes.
    # Exchange peace in G_base witch match pattern base on pattern head.
    # Ex: Graph a-b, a-c to c<-a->b pattern (0,1) (2,3) -> (0,1) (0,3) where 0, 2 sign a.
    #G_out = deepcopy(G_base) # TODO do we need always copy?
    # add all isomorphic graphs to G_out
    assert loc == (node_collection is not None)
    if loc:
        isomorphisms = find_isomorphisms_with_loc(G_base, pattern.base_graph, node_collection, node_match=node_match)
    else:
        isomorphisms = find_isomorphisms(G_base, pattern.base_graph, node_match=node_match)
        #print('iso', list(isomorphisms), G_base.nodes, pattern.base_graph.nodes)
    for iso in list(isomorphisms):
        i = 0
        conv_iso = convert_dict(iso)
        G_base = transform_G_base_on(conv_iso, pattern, G_base, node_collection)
        i += 1
        if i > 12:
            break
    return G_base


def merge_by_iso(G1, G2, P1_base, P1_head, P2_base, P2_head,
                 loc=True, node_collection=None, node_match=None):
    get_head()



if __name__ == '__main__':
    from PerceptionGraph import download_text, download_image, create_graph_from_textc
    from Visualise import plot
    text = 'adsfsdab'
    #print(text)
    #img.show()

    node_collection = NodeCollection()
    G = create_graph_from_textc(text, node_collection)
    print(G.nodes(data=True))

    nodes_data = [(1, {'sign': 'a', 'coordinate': 0, 'type': 'char'}), (2, {'sign': 'b', 'coordinate': 7, 'type': 'char'})]
    assert node_collection.find_nodes_with_same_data_in_loc(nodes_data) == {0, 7}

    a_pattern = Pattern([(0, 1), (1, 2)], [(0, 1)],
                        {0: {'type': 'char', 'sign': 'a'},
                         1: {'type': 'char', 'sign': None},
                         2: {'type': 'char', 'sign': None}})

    subG = node_collection.get_subgraph_by_pattern(G, a_pattern)
    print(subG.nodes(data=True))
    plot(subG)

    def test_isomorphism(node_match=None, base=[(0, 1), (1, 2)]):
        node_collection = NodeCollection()
        G = create_graph_from_textc('adsfsdab', node_collection)
        a_pattern = Pattern(base, [(0, 1)],{0: {'type': 'char', 'sign': 'a'},
                         1: {'type': 'char', 'sign': None}, 2: {'type': 'char', 'sign': None}})
        isomorphisms = find_isomorphisms(G, a_pattern.base_graph, node_match=node_match)
        print(list(isomorphisms))

    test_isomorphism(base=[(0, 1)],node_match=None)

    def test_get_isomorphisms_on(G_base, pattern, loc=False, node_collection=None, node_match=None):
        G_base = get_isomorphisms_on(G_base, pattern, loc=loc, node_collection=node_collection, node_match=node_match)
        for node in G_base.nodes(data=True):
            neighbour = list(G_base.neighbors(node[0]))
            print(node, neighbour)
        print(G_base.edges())
        plot(G_base)
        #nx.draw(G_out, with_labels=True)
        #plt.show()

    #test_get_isomorphisms_on(G, a_pattern, loc=False, node_match=None)#, node_collection=node_collection)


    # Test transform_on
    # More clever test go one direction and onother then choose best