import json
from DS.Logger.Vis_last_log import LogReader

def find_last_source_in_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'Source graph' in line:
            return json.loads(line)
    return None

def find_last_target_in_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'Target graph' in line:
            return json.loads(line)
    return None


def find_last_patterns_in_log(log_file):
    patterns = []
    start_id = None
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(reversed(lines)):
        if 'Source graph' in line:
            start_id = i
            break
    for i, line in enumerate(reversed(lines)):
        if 'Pattern graph' in line and i < start_id:
            patterns.append(json.loads(line))
    return patterns


# import maximum_common_induced_subgraph
from DS.PMCIS.mcs import maximum_common_induced_subgraph
import networkx as nx


if __name__ == '__main__':
    log_file = 'Log_graph.json'
    last_source = find_last_source_in_log(log_file)
    last_target = find_last_target_in_log(log_file)
    last_patterns = find_last_patterns_in_log(log_file)
    print(last_source)
    #print(last_target)
    #print(last_patterns)
    # find maximum common subgraph
    G1 = LogReader._create_graph_from_log(last_source['graph'])
    G2 = LogReader._create_graph_from_log(last_target['graph'])
    mcs = maximum_common_induced_subgraph(G1, G2)
    print(mcs)