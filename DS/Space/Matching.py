# if node/edge none
# if pass exist (path through edges type)
# if any in hiararchy



def nml(node1, node2):
    """Node match by label."""
    return node1['label'] == node2['label']

def eml(edge1, edge2):
    """Edge match by label."""
    return edge1['label'] == edge2['label']

def nmt(node1, node2):
    """Node match by type."""
    return node1['type'] == node2['type']

def emt(edge1, edge2):
    """Edge match by type."""
    return edge1['type'] == edge2['type']

def nmlt(node1, node2):
    """Node match by label and type."""
    return (node1['type'] == node2['type'] and 
            node1['label'] == node2['label'])

def emlt(edge1, edge2):
    """Edge match by label and type."""
    return (edge1['type'] == edge2['type'] and 
            edge1['label'] == edge2['label'])


def match_true_if_none(test_value, default_value):
    """Returns True if test_value is None or matches default_value."""
    # If either value is None, return True (None matches anything)
    if test_value is None:
        return True
    if default_value is None:
        return True
    # Otherwise, check for equality
    return test_value == default_value

def edge_none_match(edge1, edge2):
    """Edge match by label and type. If None, always matches."""
    return (match_true_if_none(edge1.get('label'), edge2.get('label')) and
            match_true_if_none(edge1.get('type'), edge2.get('type')))

def node_none_match(node1, node2):
    """Node match by label and type. If None, always matches."""
    return (match_true_if_none(node1.get('label'), node2.get('label')) and
            match_true_if_none(node1.get('type'), node2.get('type')))