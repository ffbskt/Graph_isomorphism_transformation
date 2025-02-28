#!/usr/bin/env python
# coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher
from collections import defaultdict
from PIL import Image
import io
import numpy as np
import random
import string
import requests
import string
import networkx as nx
from collections import deque
from copy import deepcopy


def is_None_for_any(node_data):
    for name in node_data:
        if node_data[name] is None:
            return True
    return False

def node_match(node, pattern):
    #print(123)
    for key in pattern:
        #print(pattern, pattern[key] != node.get(key))
        if pattern[key] is not None and pattern[key] != node.get(key):
            return False
    return True

def plot(G):
    # Create a dictionary of node labels, using the 'data' attribute
    labels = {}
    for node, data in G.nodes(data=True):
        if 'sign' in data:
            labels[node] = data['sign']
        else:
            print("Error: no 'sign' in node data", node, data)
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False)  # Draw nodes and edges
    nx.draw_networkx_labels(G, pos, labels=labels)  # Draw node labels

    # Show the plot
    plt.show()


def download_text(url, lower=False, remove_punctuation=False):
    response = requests.get(url)
    text = response.text

    if lower:
        text = text.lower()

    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    return text

def download_image(url, resize=None, black_and_white=False):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))

    if resize is not None:
        img = img.resize(resize)

    if black_and_white:
        img = img.convert('L')

    return img




node_collection = {}

def add_node(G, node_data=None):
    index = len(node_collection)
    node_collection[index] = node_data
    G.add_node(index, **node_data)
    return index

def create_graph_from_text(text):
    words = text.split()
    G = nx.DiGraph()
    for i, word in enumerate(words):
        add_node(G, {'word': word, 'coordinate': i, 'type': 'word'})
        if i > 0:
            G.add_edge(index-1, index)
    return G

def create_graph_from_textc(text):
    chars = list(text)
    G = nx.DiGraph()
    for i, ch in enumerate(chars):
        index = add_node(G, {'sign': ch, 'coordinate': i, 'type': 'char'})

        if i > 0:
            G.add_edge(index-1, index)
    return G

def create_graph_from_image(img):
    G = nx.DiGraph()

    for i in range(img.height):
        for j in range(img.width):
            #index = len(node_collection)
            #node_data = {'intensity': img.getpixel((j, i)), 'coordinate': (j, i), 'type': 'pixel'}
            #node_collection[index] = node_data
            #G.add_node(index, **node_data)
            index = add_node(G, {'intensity': img.getpixel((j, i)), 'coordinate': (j, i), 'type': 'pixel'})

            if i > 0:
                G.add_edge(index-img.width, index)  # Connect to the pixel above
            if j > 0:
                G.add_edge(index-1, index)  # Connect to the pixel to the left

    return G


class Pattern:
    def __init__(self, base, head, data):
        self.base_graph = self.create_graph(base, data)
        self.head_graph = self.create_graph(head, data)
        self.base_size_not_none = self.find_not_none_size(self.base_graph)

    def find_not_none_size(self, graph):
        size = 0
        for node_id, node_data  in graph.nodes(data=True):
            if not is_None_for_any(node_data):
                size += 1
        return size

    @staticmethod
    def create_graph(nodes, data):
        graph = nx.DiGraph()
        for node in nodes:
            if len(node) == 2:
                graph.add_node(node[0], **data.get(node[0], {}))
                graph.add_node(node[1], **data.get(node[1], {}))
                graph.add_edge(*node)
            elif len(node) == 1:
                graph.add_node(node[0], **data.get(node[0], {}))
        return graph

# Generate random base and head
base = [(random.randint(1,10), random.randint(1,3)) for _ in range(2)]
head = [(random.randint(1,10), random.randint(1,4)) for _ in range(3)]

# Generate random data
data = {i: {'sign': random.choice(string.ascii_lowercase), 'type': 'char'} for i in range(1, 11)}

# Create a Pattern object
pattern = Pattern(base, head, data)

# Print the base_graph and head_graph to check if they are created correctly
print('Base Graph:', base, data)
print(pattern.base_graph.nodes(data=True))
print('Head Graph:', head, data)
print(pattern.head_graph.nodes(data=True))

nx.draw(pattern.head_graph, with_labels=True)
nx.draw(pattern.base_graph, with_labels=True)
plt.show()
a

def find_isomorphisms(pattern_base, G_text, node_match=None, edge_match=None):
    return nx.algorithms.isomorphism.GraphMatcher(
        G_text, pattern_base, node_match=node_match, edge_match=edge_match).subgraph_isomorphisms_iter()




a_pattern = Pattern([(0, 1)], [(0, 1)],
                    {0: {'type': 'char', 'sign':'a'},
                     1: {'type': 'char', 'sign':None}})

def subgraph_with_neighbors(G, node_list, depth=1):
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


# get all names from loc
def get_all_names(loc):
    for key in loc.keys():
        if key.endswith('2nodes'):
            yield key[:-6]



# find all nodes with same data as in pattern base
def find_nodes_with_same_data_in_loc(pattern, loc):
    all_matched_nodes = set()
    # takde node from pattern base and find all nodes with same data in loc
    for node_id, node_data in pattern.base_graph.nodes(data=True):
        one_node_matched_nodes = set()
        for data_name in node_data:
            if is_None_for_any(node_data):
                # TODO: how work with None in one or two variables?
                # TODO: find all nodes with None ...
                continue
            nodes = loc[data_name+'2nodes'][node_data[data_name]]
            # if one_node_matched_nodes is empty, then add all nodes else find intersection
            if not one_node_matched_nodes:
                one_node_matched_nodes = set(nodes)
            else:
                one_node_matched_nodes = one_node_matched_nodes.intersection(nodes)
        all_matched_nodes = all_matched_nodes.union(one_node_matched_nodes)
    return all_matched_nodes

# function that find isomorphysms include sign of node.
# TODO: problem if all pattern nodes None or have not data
def find_isomorphisms_with_loc(G_base, pattern):
    all_matched_nodes = find_nodes_with_same_data_in_loc(pattern, loc)
    sub_graphs = subgraph_with_neighbors(G_base, all_matched_nodes, depth=pattern.base_size_not_none)
    isomorphisms = find_isomorphisms(pattern.base_graph, sub_graphs, node_match=node_match)
    # filter isomorphisms
    return isomorphisms

def convert_dict(dict):
    return {value: key for key, value in dict.items()}

#_----------------------------------------------------------------------------------------------------------------------


def transform_add_nodes(conv_iso, pattern, G_out, G_base):
    # Add nodes from pattern head to G_out
    iso_head2out = {}
    for node in pattern.head_graph.nodes():
        if node not in conv_iso:
            index = add_node(G_out, pattern.head_graph.nodes[node])
            iso_head2out[node] = index
        else:
           index = add_node(G_out, G_base.nodes[conv_iso[node]])
           iso_head2out[node] = index
    return iso_head2out

"""
def change_attw(iso, G_input, global_attention, pattern_attention):
    for node in pattern_attention['pos']:
        G_input.nodes[iso[node]]['attw'] *= 1.1
        global_attention.add(iso[node])
    for node in pattern_attention['neg']:
        G_input.nodes[iso[node]]['attw'] *= 0.9
        # remove node from global_attention
        if iso[node] in global_attention:
            global_attention.remove(iso[node])
    print('neg', G_input.nodes[1]['data'], G_input.nodes[1]['attw'])
"""

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
            #print(conv_iso[base_edge[0]], conv_iso[base_edge[1]])
            if (conv_iso[base_edge[0]], conv_iso[base_edge[1]]) in G_base.edges():
                G_base.remove_edge(conv_iso[base_edge[0]], conv_iso[base_edge[1]])
            else:
                print('remove edge no edge in base', base_edge, conv_iso[base_edge[0]], conv_iso[base_edge[1]])

def transform_add_nodes_on(conv_iso, pattern, G_base):
    # Add nodes from pattern head to G_out
    iso_head2out = {}
    for node in pattern.head_graph.nodes():
        if node not in conv_iso:
            index = add_node(G_base, pattern.head_graph.nodes[node])
            iso_head2out[node] = index
    iso_head2out.update(conv_iso)
    return iso_head2out

def transform_add_edges_on(conv_iso, pattern, G_base):
    for edge in pattern.head_graph.edges():
        if edge not in G_base.edges():
            G_base.add_edge(conv_iso[edge[0]], conv_iso[edge[1]])

#_____________----------------------------------------------------------------------------------------------------------
def transform_G_out(iso, pattern, G_out, G_base):
    iso_head2out = transform_add_nodes(iso, pattern, G_out, G_base)
    G_out.add_edges_from([(iso_head2out[node1], iso_head2out[node2]) for node1, node2 in pattern.head_graph.edges])

def transform_G_base_on(iso, pattern, G_base):
    iso_head2base = transform_add_nodes_on(iso, pattern, G_base)
    transform_add_edges_on(iso_head2base, pattern, G_base)
    transform_remove_nodes(iso_head2base, pattern, G_base)
    transform_remove_edges(iso_head2base, pattern, G_base)


print(pattern.head_graph.nodes, pattern.head_graph.edges)

def get_isomorphisms_from(G_base, pattern, loc=True):
    # Always add new nodes from base or from head
    # Take nodes from G_base match with pattern nodes.
    # Create new graph with pattern head collection and data from G_base
    # Ex: G_text a-b-c-a-b pattern (1, 0) sign a, None transfer a-b and a-c to G_out
    G_out = nx.DiGraph()
    # add all isomorphic graphs to G_out
    if loc:
        isomorphisms = find_isomorphisms_with_loc(G_base, pattern)
    else:
        isomorphisms = find_isomorphisms(pattern.base_graph, G_base, node_match=node_match)
    #print(list(isomorphisms), G_base.nodes, pattern.base_graph.nodes)
    i = 0
    for iso in isomorphisms:
        conv_iso = convert_dict(iso)
        transform_G_out(conv_iso, pattern, G_out, G_base)
        i += 1
        if i > 10:
            break
    return G_out

def get_isomorphisms_on(G_base, pattern, loc=True):
    # Instead of get_isomorphisms_from if nodes already exist leave it or remove.
    # Edit nodes in G_base match with pattern nodes.
    # Exchange peace in G_base witch match pattern base on pattern head.
    # Ex: Graph a-b, a-c to c<-a->b pattern (0,1) (2,3) -> (0,1) (0,3) where 0, 2 sign a.
    G_out = deepcopy(G_base) # TODO do we need always copy?
    # add all isomorphic graphs to G_out
    if loc:
        isomorphisms = find_isomorphisms_with_loc(G_base, pattern)
    else:
        isomorphisms = find_isomorphisms(pattern.base_graph, G_base, node_match=node_match)
    #print(list(isomorphisms), G_base.nodes, pattern.base_graph.nodes)
    for iso in list(isomorphisms):
        i = 0
        conv_iso = convert_dict(iso)
        transform_G_base_on(conv_iso, pattern, G_base)
        i += 1
        if i > 12:
            break
    return G_out






#count_pattern = Pattern([(0)], [(0, 0)], {0: {'type': 'char', 'sign':'a'}, 1: {'type': 'char', 'sign':None}})

# Create a subgraph of G with the first 20 nodes
#G_sub = G_text.subgraph(list(G_text.nodes)[:20])

# Transform the subgraph by the pattern
#G_transformed = transform_graph_by_pattern(G_sub, pattern)

# Print the nodes of the transformed graph
#print(G_transformed.nodes(data=True))

if __name__ == "__main__":
    # Test for 'download_text' function
    url = 'http://example.com'
    text = download_text(url, lower=True, remove_punctuation=True)
    # print(text[:100])  # Print the first 100 characters of the text

    # Test for 'download_image' function
    url = 'https://platinumlist.net/guide/wp-content/uploads/2023/03/IMG-worlds-of-adventure.webp'
    img = download_image(url, resize=(100, 100), black_and_white=False)
    # img.show()  # Display the image

    # Test for 'create_graph_from_text' function
    G_text = create_graph_from_textc(text)
    # print(G_text.nodes(data=False))  # Print the nodes of the graph

    # Test for 'create_graph_from_image' function
    G_image = create_graph_from_image(img)


    # print(G_image.nodes(data=False))  # Print the nodes of the graph
    G_base = nx.DiGraph()
    G_base.add_edge(0, 1)
    G_base.add_edge(2, 1)
    # Find isomorphisms
    isomorphisms = find_isomorphisms(pattern.base_graph, G_image)
    isomorphisms2 = find_isomorphisms(G_base, G_image)
    print("Isomorphism in image", isomorphisms, isomorphisms2)

    # need collection for all node data: sign, type, coordinate, intensity
    type2nodes = defaultdict(list)
    sign2nodes = defaultdict(list)
    intensity2nodes = defaultdict(list)
    coordinate2nodes = defaultdict(list)
    loc = {'type2nodes': type2nodes, 'sign2nodes': sign2nodes, 'intensity2nodes': intensity2nodes,
           'coordinate2nodes': coordinate2nodes}
    # TODO: need to add intervals for each graph {self: [0, 127], ...}
    graph2nodes_intervals = defaultdict(list)  # intervals of nodes numbers for each graph

    for node_id, node_data in node_collection.items():
        for data_name in ['type', 'sign', 'intensity', 'coordinate']:
            if data_name in node_data:
                loc[data_name + '2nodes'][node_data[data_name]].append(node_id)


    # for iso in isomorphisms:
    #    print(iso)
    #    break

    # Example usage
    def test_subgraph_with_neighbors():
        G = G_image
        node_list = coordinate2nodes[(1, 1)] + coordinate2nodes[(3, 3)]
        G = G_text
        node_list = sign2nodes['a'][:3]
        depth = 1
        print(node_list)
        H = subgraph_with_neighbors(G, node_list, depth)
        nx.draw(H, with_labels=True)
        plt.show()

    #test_subgraph_with_neighbors()

    def test_find_nodes_with_same_data_in_loc():
        filtered_nodes = find_nodes_with_same_data_in_loc(a_pattern, loc)
        print(a_pattern.base_graph.nodes(data=True), filtered_nodes)
        for node_id in filtered_nodes:
            print(G_text.nodes[node_id])

    #test_find_nodes_with_same_data_in_loc()
# Filtered nodes is equal graphs that isomorphic some pattern - [sign, type, coordinate, intensity]
# And we use isomorphism then isomorphysm again. We could just memorize it.
# What is sign: a? It is general isomorphysm to all images looks like a and to text sign a.

    def test_get_isomorphisms_from(G_base, pattern):
        G_out = get_isomorphisms_from(G_base, pattern)
        for node in G_out.nodes(data=True):
            neighbour = list(G_out.neighbors(node[0]))
            print(node, neighbour)
        plot(G_out)
        #nx.draw(G_out, with_labels=True)
        #plt.show()

    # test_get_isomorphisms_from(G_text, a_pattern)
    """
    G = nx.DiGraph()
    # Add nodes with data to the graph
    edge_patern = Pattern([(0,), (1,)], [(0, 1)], {0: {'type': 'char', 'sign': 'a'}, 1: {'type': 'char', 'sign': 'a'}})

    data = {1: {'type': 'char', 'sign': 'a'},
            2: {'type': 'char', 'sign': 'b'},
            3: {'type': 'char', 'sign': 'a'},
            4: {'type': 'char', 'sign': 'b'},
            5: {'type': 'char', 'sign': 'a'},
            6: {'type': 'char', 'sign': 'c'}
            }
    for i in range(1, 7):
        G.add_node(i, **data.get(i, {}))

    # graph.add_node(node[0], **data.get(node[0], {}))
    # G.add_node(1, {'type': 'char', 'sign':'a'})

    # Add edges to the graph
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    G.add_edge(5, 6)

    #G_out = get_isomorphisms_from(G, edge_patern, loc=False)
    # print(G_out.nodes(data=True))
    # print(a_pattern.base_graph.nodes(data=True), G.nodes(data=True))
    # plot(G)
    # plot(edge_patern.base_graph)
    #plot(G_out)

    # get_isomorphisms_on(G, edge_patern, loc=False)
    #G_out = get_isomorphisms_from2(G, edge_patern, loc=False)
    """
    G_out = get_isomorphisms_from(G_text, a_pattern)
    plot(G_out)
    edge_pattern = Pattern([(0,1), (2,3)], [(0, 1), (0,3)], {0: {'type': 'char', 'sign': 'a'}, 1: {'type': 'char', 'sign': None},
                                                      2: {'type': 'char', 'sign': 'a'}, 3: {'type': 'char', 'sign': None}})
    #edge_pattern = Pattern([(0,1)], [(0,), (1,)], {0: {'type': 'char', 'sign': 'a'}, 1: {'type': 'char', 'sign': None}})
    #plot(edge_pattern.head_graph)
    get_isomorphisms_on(G_out, edge_pattern, loc=False)
    print(G_out.nodes(data=True))
    plot(G_out)