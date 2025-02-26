from graph_isomorphism import Graph, check_isomorphism

def main():
    # Create two isomorphic graphs
    # Graph 1: Square
    g1 = Graph()
    g1.add_edge(1, 2)
    g1.add_edge(2, 3)
    g1.add_edge(3, 4)
    g1.add_edge(4, 1)

    # Graph 2: Square with different vertex labels
    g2 = Graph()
    g2.add_edge(5, 6)
    g2.add_edge(6, 7)
    g2.add_edge(7, 8)
    g2.add_edge(8, 5)

    # Check if the graphs are isomorphic
    result = check_isomorphism(g1, g2)
    print("Graphs are isomorphic:", result)

    # Print graph information
    print("\nGraph 1:")
    print("Vertices:", g1.get_vertices())
    print("Edges:", g1.get_edges())
    
    print("\nGraph 2:")
    print("Vertices:", g2.get_vertices())
    print("Edges:", g2.get_edges())

if __name__ == "__main__":
    main() 