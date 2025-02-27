import networkx as nx
from graph_visualization import GraphVisualizer

def create_example_graph():
    """Create an example graph with different node and edge types."""
    G = nx.Graph()
    
    # Add nodes with different types and labels
    nodes = [
        (1, {'type': 'type1', 'label': 'Circle 1'}),
        (2, {'type': 'type2', 'label': 'Square 1'}),
        (3, {'type': 'type3', 'label': 'Triangle 1'}),
        (4, {'type': 'type1', 'label': 'Circle 2'}),
        (5, {'type': 'type2', 'label': 'Square 2'})
    ]
    G.add_nodes_from(nodes)
    
    # Add edges with different types and labels
    edges = [
        (1, 2, {'type': 'black', 'label': 'Edge 1'}),
        (2, 3, {'type': 'white', 'label': 'Edge 2'}),
        (3, 4, {'type': 'black', 'label': 'Edge 3'}),
        (4, 5, {'type': 'white', 'label': 'Edge 4'}),
        (5, 1, {'type': 'black', 'label': 'Edge 5'})
    ]
    G.add_edges_from(edges)
    
    return G

def main():
    # Create example graph
    G = create_example_graph()
    
    # Create visualizer
    visualizer = GraphVisualizer()
    
    # Draw the graph
    visualizer.draw_graph(G, title="Example Graph with Different Node and Edge Types")
    
    # Show the graph
    visualizer.show()
    
    # Optionally save the graph
    visualizer.save("example_graph.png")

if __name__ == "__main__":
    main() 