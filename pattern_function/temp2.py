import networkx as nx
from GraphTransformation import Pattern, NodeCollection, plot, debug_print, \
    find_isomorphisms, node_match, find_isomorphisms_with_loc, convert_dict, add_node
from PerceptionGraph import download_text, download_image, create_graph_from_textc
import random
from collections import defaultdict
from collections import deque
from Visualise import plot, col_plot
from copy import deepcopy
import inspect


def func_show_args(func):
    sig = inspect.signature(func)
    return [param.name for param in sig.parameters.values()]


def sign_to_iso(iso, P, G2):
    print('sign_to_iso------------------', iso, G2.nodes)
    for key in iso:
        if 'sign' in G2.nodes[key]:
            print(G2.nodes[key]['sign'], ':', iso[key], end=' ')
        else:
            print(key, ':', iso[key], end=' ')
    print()

text = 'adsfsdab'
#print(text)
#img.show()

#"""
node_collection = NodeCollection()
G = create_graph_from_textc(text, node_collection)
print(G.nodes(data=True))
debug_print(G)


pattern = Pattern([(1, 2),], [(1,4),(4, 2)],
                  {1: {'sign': 'a', 'type': 'char'},
                   2: {'sign': None, 'type': 'char'},
                   4: {'sign': '_', 'type': 'char'},})
#"""
def get_isomorphism(G, pattern, loc=False, node_collection=None, node_match=None):
    assert loc == (node_collection is not None)
    if loc:
        isomorphisms = find_isomorphisms_with_loc(G, pattern, node_collection, node_match=node_match)
    else:
        isomorphisms = find_isomorphisms(G, pattern, node_match=node_match)
        # print('iso', list(isomorphisms), G_base.nodes, pattern.base_graph.nodes)
    return isomorphisms

## def remove_repetition(): if iso have "a b c" then next iso should be "d e f" not bcd
## real copy cut shoud find exactly last letter with link to "start node"


def transfer_data(G1, G2, isomorphism=None):
    # Transfer data from G1 to G2 by isomorphisms
    isomorphism = isomorphism or trivial_iso(G1, G2)
    for node in isomorphism:
        G2.nodes[isomorphism[node]].update(G1.nodes[node])

def transfer_edges(G1, G2, isomorphism=None):
    # Go through each node in the isomorphisms and add edges to G2
    isomorphism = isomorphism or trivial_iso(G1, G2)
    for node in isomorphism:
        if node in G1:
            neighbors = G1.neighbors(node)
            for neighbor in neighbors:
                if neighbor in isomorphism:
                    G2.add_edge(isomorphism[node], isomorphism[neighbor])

# + def nx.merge_graphs Open question how to do that.
def merge(G1, G2):
    corent_iso = {}
    for node, data in G1.nodes(data=True):
        index = add_node(G2, node_collection, data)
        corent_iso[node] = index
    for edges in G1.edges:
        G2.add_edge(corent_iso[edges[0]], corent_iso[edges[1]])

def merge_with_replace(G1, G2, isomorphism=None, remove_old_conections=True):
    # Merge G1 to G2 with replace nodes from isomorphisms
    isomorphism = isomorphism or trivial_iso(G1, G2)
    conv_iso = convert_dict(isomorphism)
    for node, data in G1.nodes(data=True):
        if node in conv_iso:
            #print('replace', node, data, conv_iso[node])
            G2.nodes[conv_iso[node]].update(data)
        else:
            index = add_node(G2, node_collection, data)
            conv_iso[node] = index
    if remove_old_conections:
        list_of_nodes = list(set(isomorphism.keys()) & set([conv_iso[n] for n in G1.nodes]))
        for i in range(len(list_of_nodes)):
            for j in range(len(list_of_nodes)):
                if G2.has_edge(list_of_nodes[i], list_of_nodes[j]):
                    G2.remove_edge(list_of_nodes[i], list_of_nodes[j])
    for edges in G1.edges:
        G2.add_edge(conv_iso[edges[0]], conv_iso[edges[1]])
    #plot(G2)


def trivial_iso(G1, G2):
    # Return trivial isomorphism. itersection of nodes G1, G2.
    nodes = G1.nodes & G2.nodes
    return {i:i for i in nodes}

class ListGenerator:
    def __init__(self, list_of_nodes):
        self.list_of_nodes = list_of_nodes

    def __iter__(self):
        # pop first element from self.list_of_nodes
        return self

    def __next__(self):
        return self.list_of_nodes.pop(0)

    def __len__(self):
        return len(self.list_of_nodes)


def remove_repeated_nodes(iso_map_list):
    seen_keys = set()
    new_list = []
    for iso in iso_map_list:
        if not seen_keys.intersection(iso.keys()):
            new_list.append(iso)
            seen_keys.update(iso.keys())
    return new_list


# No same but without loop.
# We show parallel computation. But we have limited input and transfer capacity. So thats why we need sequential operations.

def f_start(G, copyG, start_copy_pattern, start_past_pattern):
    iso = get_isomorphism(G, start_copy_pattern.base_graph, loc=False, node_collection=None, node_match=node_match)
    iso = ListGenerator(list(iso)) #remove_repeated_nodes(list(iso)))
    first_iso = next(iso)
    transfer_data(G, start_copy_pattern.base_graph, isomorphism=first_iso)
    transfer_data(start_copy_pattern.base_graph, start_past_pattern.head_graph)
    start_copy_pattern.clean()
    merge(start_past_pattern.head_graph, copyG)
    return iso

def f_continue (G, copyG, continue_copy_pattern, continue_past_pattern, iso):
    if len(iso) == 0:
        return None
    isomorphism = next(iso)
    transfer_data(G, continue_copy_pattern.base_graph, isomorphism=isomorphism)
    # debug_print(continue_copy_pattern.base_graph)
    if continue_copy_pattern.head_graph: ## Or if head is empty head_graph = base_graph
        transfer_data(continue_copy_pattern.base_graph, continue_copy_pattern.head_graph)
        debug_print(continue_copy_pattern.head_graph)
        transfer_data(continue_copy_pattern.head_graph, continue_past_pattern.head_graph)
    else:
        transfer_data(continue_copy_pattern.base_graph, continue_past_pattern.head_graph)
    iso2 = get_isomorphism(copyG, continue_past_pattern.base_graph, loc=False, node_collection=None, node_match=node_match)
    #continue_copy_pattern.clean()
    i2 = list(iso2)
    if i2: # if past graph already in copyG
        merge_with_replace(continue_past_pattern.head_graph, copyG, i2[0], remove_old_conections=True)
    return iso

def f_end(iso, cur_node_pointer, G_computation):
    if len(iso) == 0:
        print('end', iso, cur_node_pointer, i3, a9)  # G_computation.nodes[cur_node_pointer]['data'])
        # Get node with sign cur_node_pointer from G_computation
        for node in G_computation.successors(cur_node_pointer):
            if G_computation.nodes[node]['sign'] == 'cur_node_pointer':
                G_computation.nodes[node]['data'] = None
        return None
    return cur_node_pointer






#----- Computation graph.-------------------------------------------------------------------------------------------
# Сomputational graph handling comunication between graphs by patterns.

communication_graph = nx.DiGraph()
i1 = add_node(communication_graph, node_collection, node_data={'sign': 'start', 'type': 'control_function', 'function': f_start})
i2 = add_node(communication_graph, node_collection, node_data={'sign': 'continue', 'type': 'control_function', 'function': f_continue})
i3 = add_node(communication_graph, node_collection, node_data={'sign': 'end', 'type': 'control_function', 'function': f_end})
a1 = add_node(communication_graph, node_collection, node_data={'sign': 'G', 'type': 'arg'})
a2 = add_node(communication_graph, node_collection, node_data={'sign': 'copyG', 'type': 'arg'})
a3 = add_node(communication_graph, node_collection, node_data={'sign': 'start_copy_pattern', 'type': 'arg'})
a4 = add_node(communication_graph, node_collection, node_data={'sign': 'start_past_pattern', 'type': 'arg'})
a5 = add_node(communication_graph, node_collection, node_data={'sign': 'continue_copy_pattern', 'type': 'arg'})
a6 = add_node(communication_graph, node_collection, node_data={'sign': 'continue_past_pattern', 'type': 'arg'})
a7 = add_node(communication_graph, node_collection, node_data={'sign': 'iso', 'type': 'arg'})
a8 = add_node(communication_graph, node_collection, node_data={'sign': 'G_computation', 'type': 'arg'})
a9 = add_node(communication_graph, node_collection, node_data={'sign': 'cur_node_pointer', 'type': 'arg', 'data': None})
# curent_node_pointer is always exist and only one. Except of other arguments.
# Even G_computation could be changed between two computations graphs. (Is it possible?)
# Q: we need change it in data_collection??? Do we really need args graph + data collection mechanism?
#a8 = add_node(communication_graph, node_collection, node_data={'sign': 'iso', 'type': 'return'})

cur_node_pointer = None
communication_graph.add_edges_from([(i1,i2), (i2, i3), (i3,i2)]) # add i3 out.
# Argument graph. Better if we split it on simple functions and pattern that conect argumets with functions.
# But this mechanism we could change on labels. TODO Patern <-> Label exchange mechanism.
# TODO make a sequence of arguments graph a1-a2-a3-a4, a1-a2-a5-a6-a7.
communication_graph.add_edges_from([(a1,i1), (a2,i1), (a3,i1), (a4,i1), (i1, a7),
                                    (a1,i2), (a2,i2), (a5,i2), (a6,i2), (a7,i2), (i2, a7),
                                    (a7, i3), (a8, i3), (a9, i3), (i3, a9)])
#plot(communication_graph)
# TODO here is two options labeling or patterns. Pattern is a key that links to dour and connect it.
#  Inside Pattern base1 exactly how look this door - the same as a label.
#  But between patterns easy to find nearest and order of nearest.
#  So we just generate patterns and try it to all possible doors (functions and arguments) and memorize the path.
# After find a path we can create higher hirarchy pattern.
pattern_collection = [Pattern([(1,2),], [()], {}),
                      Pattern([(3,),], [(1,2), (2,3)], {3: {"sign": "start", "type": "control"}}),
                      Pattern([(1,2),], [()], {}),
                      Pattern([(4,5),(5,3),], [(4, 5), (5,2), (2,3)], {3: {"sign": "start", "type": "control"}})
                      ]

text = 'adsfsdab'
#plot(pattern_collection[0].base_graph)
#plot(pattern_collection[1].base_graph)
#plot(pattern_collection[2].base_graph)
#plot(pattern_collection[3].base_graph)

node_collection = NodeCollection()
G = create_graph_from_textc(text, node_collection)
G_copy = nx.DiGraph()
col_plot(communication_graph)

# TODO the same as graph. Just we need trivial isomorphism.
#  Args in graph could link to different data nodes in data_nodes in data_collection_graph, but here it link trivial.
data_collection = {'G_computation':communication_graph, 'cur_node_pointer': cur_node_pointer,
                   'G': G, 'copyG': G_copy,
                   'start_copy_pattern': pattern_collection[0], 'start_past_pattern': pattern_collection[1],
                   'continue_copy_pattern': pattern_collection[2], 'continue_past_pattern': pattern_collection[3],
                   'iso': None
                   }

def connect_arg_with_data(G_computation, data_collection):
    # for all args in G_computation
    # This function will be replaced by descision graph, that will choose what data link with arg.
    # TODO open question if we need build own connection data-arg-functions or arg+data-function?
    for node in G_computation.nodes:
        if G_computation.nodes[node]['type'] == 'arg':
            G_computation.nodes[node]['data'] = data_collection[G_computation.nodes[node]['sign']]

def return_data_to_arg(G_computation, data_collection, curent_node, return_data):
    # TODO if return_data is list, we need pattern to break it on args.
    for node in G_computation.successors(curent_node):
        if G_computation.nodes[node]['type'] == 'arg':
            #print('return_data_to_arg', G_computation.nodes[node], return_data)
            G_computation.nodes[node]['data'] = return_data



def create_args(function_args, G_computation, current_node):
    # TODO open question if we need build own connection
    args = {}
    args_nodes = [arg for arg in G_computation.predecessors(current_node) if G_computation.nodes[arg]['type'] == 'arg']
    for node in args_nodes:
        args[G_computation.nodes[node]['sign']] = G_computation.nodes[node]['data']
    #print(args, G_computation.nodes[a9]['data'])
    return args

def step(G_computation, current_node, data_collection):
    # choose what data link with arg. Write it in data. Maybe clean will be needed.
    # get function from data in current node:
    function = G_computation.nodes[current_node]['function'] # take function
    args_name = func_show_args(function) #(G, copyG, pattern_collection[0], pattern_collection[1])
    args = create_args(args_name, G_computation, current_node)
    return_value = function(**args)
    return_data_to_arg(G_computation, data_collection, current_node, return_value)
    # Get a list of the current node's successors in the computational graph
    # change data if function return something to successors.
    successors = list(G_computation.successors(current_node))
    next_functions = []
    for node in successors:
        if 'return' in G_computation.nodes[node]['type']:
            G_computation.nodes[node]['data'] = return_value
        # if 'control' in G_computation.nodes[node]['type']:
        #    next_functions.append(node)

    # If the current node has no successors, return it as the next node
    #if not next_functions:
    #    return None
    # Choose a successor randomly to be the next node
    #next_node = i2#random.choice(next_functions)
    #return next_node

# Handling copy
"""
connect_arg_with_data(communication_graph, data_collection)
step(communication_graph, i1, data_collection)
step(communication_graph, i2, data_collection)
step(communication_graph, i2, data_collection)
step(communication_graph, i2, data_collection)
step(communication_graph, i2, data_collection)
step(communication_graph, i2, data_collection)
step(communication_graph, current_node=i2, data_collection=data_collection)
#print(list(communication_graph.nodes[a7]['data']))
communication_graph.nodes[a9]['data'] = i3
step(communication_graph, current_node=i3, data_collection=data_collection)
#print(communication_graph.nodes[a9])
plot(G_copy)
"""
# G_computation graph upper level

def chose_next_function_node(G_computation, current_node):
    successors = list(G_computation.successors(current_node))
    next_functions = []
    for node in successors:
        if 'control' in G_computation.nodes[node]['type']:
            next_functions.append(node)
    if not next_functions:
        return None
    next_function_node = random.choice(next_functions)
    return next_function_node

run = 30
def compute(run, G_computation, data_collection, start_pointer, visual=lambda x: x):
    # Choose a random node from the computational graph to start
    current_node = start_pointer  # random.choice(list(G_computation.nodes))
    visual(1) ## ?? add information about step i, pointer,
    # Some times should do this (maybe in loop):
    connect_arg_with_data(G_computation, data_collection) # refresh data in nodes. ?? Is it need??
    G_computation.nodes[a9]['data'] = current_node # write where is pointer
    # Perform the step function in a loop for the given number of runs
    for i in range(run):
        #print(123, current_node, 1000,  G_computation.nodes[a9]['data'])
        if current_node is None:
            #or choose some new tasks
            return
        step(G_computation, current_node, data_collection) # why we need data collection??
        # if function work and do not change pointer we continue compute
        if current_node == G_computation.nodes[a9]['data']:
            current_node = chose_next_function_node(G_computation, current_node)
            G_computation.nodes[a9]['data'] = current_node
        else:
            # if function change pointer - it is mean only end function and process is finish
            # So we go to new pointer
            current_node = G_computation.nodes[a9]['data']




compute(run, communication_graph, data_collection, start_pointer=i1)

plot(data_collection['copyG'])


# _______________step 2 create copy for only odd letters_______________________
"""
For this task we need only change patterns in communication graph. Change only 2 collection below.
"""
text = 'adsfsdab'
pattern_collection = [Pattern([(1,2),], [()], {}),
                      Pattern([(3,),], [(1,2), (2,3)], {3: {"sign": "start", "type": "control"}}),
                      Pattern([(1,2),], [()], {}),
                      Pattern([(4,5),(5,3),], [(5,2), (2,3)], {3: {"sign": "start", "type": "control"}})
                      ]
data_collection = {'G_computation':communication_graph, 'cur_node_pointer': cur_node_pointer,
                   'G': G, 'copyG': G_copy,
                   'start_copy_pattern': pattern_collection[0], 'start_past_pattern': pattern_collection[1],
                   'continue_copy_pattern': pattern_collection[2], 'continue_past_pattern': pattern_collection[3],
                   'iso': None
                   }

data_collection['start_copy_pattern'] = Pattern([(1,2),(2,3)], [()], {})
data_collection['start_past_pattern']  = Pattern([(6,),], [(1,2), (2,6)], {6: {"sign": "start", "type": "control"}})
data_collection['continue_copy_pattern'] = Pattern([(1,2), (2,3)], [(1,3)], {})
data_collection['continue_past_pattern'] = Pattern([(4,5),(5,6),], [(5,3), (3,6)], {6: {"sign": "start", "type": "control"}})
data_collection['copyG'] = nx.DiGraph()
data_collection['iso'] = None
data_collection['cur_node_pointer'] = None

#plot(data_collection['copyG'])
compute(run, communication_graph, data_collection, start_pointer=i1, visual=lambda x: x)

plot(data_collection['copyG'])