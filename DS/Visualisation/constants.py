"""Constants and configurations for graph visualization."""

# Node shapes available in NetworkX
NODE_SHAPES = {
    'circle': 'o',
    'square': 's',
    'triangle': '^',
    'triangle_down': 'v',
    'diamond': 'd',
    'pentagon': 'p',
    'hexagon': 'h',
    'plus': 'P',
    'star': '*',
    'octagon': '8'
}

# Standard colors from GraphViz
GRAPHVIZ_COLORS = {
    # Basic colors
    'black': '#000000',
    'white': '#FFFFFF',
    'red': '#FF0000',
    'green': '#00FF00',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'magenta': '#FF00FF',
    'cyan': '#00FFFF',
    
    # GraphViz default colors
    'lightgray': '#D3D3D3',
    'gray': '#808080',
    'darkgray': '#A9A9A9',
    'lightblue': '#ADD8E6',
    'lightgreen': '#90EE90',
    'lightcoral': '#F08080',
    'lightyellow': '#FFFFE0',
    'lightpink': '#FFB6C1',
    'tan': '#D2B48C',
    'brown': '#A52A2A',
    'violet': '#EE82EE',
    'navy': '#000080',
    'maroon': '#800000',
    'forestgreen': '#228B22'
}

# Default node type configurations
DEFAULT_NODE_CONFIGS = {
    'H': {
        'shape': NODE_SHAPES['triangle'],
        'color': GRAPHVIZ_COLORS['lightblue'],
        'size': 500
    },
    'T': {
        'shape': NODE_SHAPES['triangle'],
        'color': GRAPHVIZ_COLORS['lightgreen'],
        'size': 500
    },
    'L': {
        'shape': NODE_SHAPES['triangle'],
        'color': GRAPHVIZ_COLORS['lightyellow'],
        'size': 500
    },
    'G': {
        'shape': NODE_SHAPES['circle'],
        'color': GRAPHVIZ_COLORS['cyan'],
        'size': 500
    },
    'P': {
        'shape': NODE_SHAPES['triangle'],
        'color': GRAPHVIZ_COLORS['lightpink'],
        'size': 500
    },
    '1': {
        'shape': NODE_SHAPES['circle'],
        'color': GRAPHVIZ_COLORS['gray'],
        'size': 500
    },
    '0': {
        'shape': NODE_SHAPES['circle'],
        'color': GRAPHVIZ_COLORS['lightgray'],
        'size': 500
    },
    'reward_outside': {
        'shape': NODE_SHAPES['star'],
        'color': GRAPHVIZ_COLORS['red'],
        'size': 500
    },
    'reward_inside': {
        'shape': NODE_SHAPES['star'],
        'color': GRAPHVIZ_COLORS['lightpink'],
        'size': 300
    },
    'default': {
        'shape': NODE_SHAPES['square'],
        'color': GRAPHVIZ_COLORS['lightgray'],
        'size': 500
    }
}

# Default edge type configurations
DEFAULT_EDGE_CONFIGS = {
    'h0': { # common horizontal edge
        'color': GRAPHVIZ_COLORS['black'],
        'width': 1.5
    },
    'white': {
        'color': GRAPHVIZ_COLORS['lightgray'],
        'width': 1.5
    },
    'default': {
        'color': GRAPHVIZ_COLORS['gray'],
        'width': 1.0
    },
    'h': {
        'color': GRAPHVIZ_COLORS['blue'],
        'width': 1.5
    },
    'hh': {
        'color': GRAPHVIZ_COLORS['blue'],
        'width': 2.0
    },
    'l': {
        'color': GRAPHVIZ_COLORS['green'],
        'width': 0.8
    },
    't': { # time edge
        'color': GRAPHVIZ_COLORS['red'],
        'width': 1.5
    }
}

# Layout configurations
LAYOUT_CONFIGS = {
    'spring': {
        'k': 1,        # Optimal distance between nodes
        'iterations': 50  # Number of iterations to compute layout
    },
    'circular': {},
    'random': {
        'seed': 42  # For reproducibility
    },
    'shell': {},
    'spectral': {},
    'kamada_kawai': {},
    'planar': {}
} 