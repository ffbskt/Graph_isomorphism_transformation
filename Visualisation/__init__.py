"""VisG - Graph Visualization Package"""

from .visg import VisG
from .constants import (
    NODE_SHAPES,
    GRAPHVIZ_COLORS,
    DEFAULT_NODE_CONFIGS,
    DEFAULT_EDGE_CONFIGS,
    LAYOUT_CONFIGS
)
from .layouts import grid_square_layout, HamiltonianLayout

__all__ = [
    'VisG',
    'NODE_SHAPES',
    'GRAPHVIZ_COLORS',
    'DEFAULT_NODE_CONFIGS',
    'DEFAULT_EDGE_CONFIGS',
    'LAYOUT_CONFIGS',
    'grid_square_layout',
    'HamiltonianLayout'
]

__version__ = '0.1.0' 