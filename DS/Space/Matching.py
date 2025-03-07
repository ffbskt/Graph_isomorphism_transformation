# if node/edge none
# if pass exist (path through edges type)
# if any in hiararchy

def much_ture_if_none(test_value, default_value):
    return True if test_value is None else test_value == default_value


def nml(self, node1, node2):
    """Node match by label."""
    return node1['label'] == node2['label']

def eml(self, edge1, edge2):
    """Edge match by label."""
    return edge1['label'] == edge2['label']

def nmt(self, node1, node2):
    """Node match by type."""
    return node1['type'] == node2['type']

def emt(self, edge1, edge2):
    """Edge match by type."""
    return edge1['type'] == edge2['type']

def nmlt(self, node1, node2):
    """Node match by label and type."""
    return (node1['type'] == node2['type'] and 
            node1['label'] == node2['label'])

def emlt(self, edge1, edge2):
    """Edge match by label and type."""
    return (edge1['type'] == edge2['type'] and 
            edge1['label'] == edge2['label'])

def edge_param_match(self, edge, G=None, elabel=None, etype=None):
        if G is None:
            G = self.G
        if (much_ture_if_none(elabel, G.edges[edge]['label']) 
            and much_ture_if_none(etype, G.edges[edge]['type'])):
            return True
        return False
    

def node_param_match(self, node, G=None, label=None, type=None):
        if G is None:
            G = self.G
        if (much_ture_if_none(label, G.nodes[node]['label']) 
            and much_ture_if_none(type, G.nodes[node]['type'])):
            return True
        return False
