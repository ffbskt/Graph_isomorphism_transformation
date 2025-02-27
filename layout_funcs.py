"""
Layout functions for graph visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def grid_square_layout(G):
    """
    Create a grid square layout for the graph where:
    - Column nodes ('Cj') are placed in the top row
    - Row nodes ('Ri') are placed in the leftmost column
    - Pixel nodes ('p_i_j') are placed in a grid according to their indices
    - Special nodes ('s', 'b') and class nodes ('c_k') are placed strategically
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to create layout for
        
    Returns:
    --------
    dict
        Dictionary of positions {node: (x, y)}
    """
    pos = {}
    max_i = max_j = 0
    
    # First pass to find dimensions
    for node in G.nodes():
        if node.startswith('p_'):
            _, i, j = node.split('_')
            max_i = max(max_i, int(i))
            max_j = max(max_j, int(j))
    
    # Grid size with margins
    grid_margin = 1
    total_width = max_j + 3 + (2 * grid_margin)  # +3 for row labels and margins
    total_height = max_i + 3 + (2 * grid_margin)  # +3 for column labels and margins
    
    # Place special nodes
    pos['s'] = (0, total_height - 1)  # Start node in top-left
    pos['b'] = (total_width - 1, total_height - 1)  # Boundary node in top-right
    
    # Place column nodes (Cj) in top row
    for node in G.nodes():
        if node.endswith('j'):
            j = int(node[:-1])
            pos[node] = (j + grid_margin + 1, total_height - 2)
    
    # Place row nodes (Ri) in leftmost column
    for node in G.nodes():
        if node.endswith('i'):
            i = int(node[:-1])
            pos[node] = (grid_margin, total_height - 3 - i)
    
    # Place pixel nodes (p_i_j) in grid
    for node in G.nodes():
        if node.startswith('p_'):
            _, i, j = node.split('_')
            i, j = int(i), int(j)
            pos[node] = (j + grid_margin + 1, total_height - 3 - i)
    
    # Place class nodes (c_k) in bottom row
    class_nodes = [n for n in G.nodes() if n.startswith('c_')]
    for idx, node in enumerate(class_nodes):
        spacing = total_width / (len(class_nodes) + 1)
        pos[node] = (spacing * (idx + 1), 0)
    
    return pos 

#------------------------Hamiltonian Layout--------------------------------

class HamiltonianLayout:
    def __init__(self, G, dim_final=2, dim_medium=10, learning_rate=0.01):
        """
        Initialize Hamiltonian layout with medium dimensional space optimization.
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph to layout
        dim_final : int
            Final dimension of the layout (usually 2 for visualization)
        dim_medium : int
            Dimension of the medium space for optimization
        """
        self.G = G
        self.dim_final = dim_final
        self.dim_medium = dim_medium
        # Initialize random positions in medium dimension
        self.pos = {node: np.random.randn(dim_medium) for node in G.nodes()}
        
        # Constants for Hamiltonian
        self.k_r = 1.0  # Repulsive force constant
        self.k_a = 1.0  # Attractive force constant
        self.learning_rate = learning_rate
        self.tolerance = 1e-3
        self.history = []  # Store Hamiltonian values during optimization
        
    def distance(self, pos1, pos2):
        """Calculate distance between two position vectors."""
        return np.linalg.norm(pos1 - pos2) + 1e-9
    
    def hamiltonian(self, positions):
        """Calculate system Hamiltonian."""
        H = 0
        nodes = list(positions.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                d = self.distance(positions[node1], positions[node2])
                # Repulsive term
                H += self.k_r / d
                # Attractive term (only for connected nodes)
                if self.G.has_edge(node1, node2):
                    H += self.k_a * (1.0 - d)**2
        return H
    
    def gradient(self, positions):
        """Calculate gradient for each node."""
        grad = {node: np.zeros(self.dim_medium) for node in positions}
        nodes = list(positions.keys())
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    node1, node2 = nodes[i], nodes[j]
                    r_ij = positions[node1] - positions[node2]
                    d = self.distance(positions[node1], positions[node2])
                    
                    # Repulsive force
                    repulsive_grad = -self.k_r * r_ij / d**3
                    
                    # Attractive force (only for connected nodes)
                    attractive_grad = np.zeros_like(r_ij)
                    if self.G.has_edge(node1, node2):
                        attractive_grad = -2 * self.k_a * (1.0 - d) * r_ij / d
                    
                    grad[node1] += repulsive_grad + attractive_grad
        
        return grad
    
    def optimize(self, max_iterations=1000):
        """Optimize node positions using gradient descent."""
        prev_H = float('inf')
        pbar = tqdm(range(max_iterations), desc="Optimizing layout")
        
        for _ in pbar:
            # Calculate gradient
            grad = self.gradient(self.pos)
            
            # Update positions
            for node in self.pos:
                self.pos[node] -= self.learning_rate * grad[node]
            
            # Calculate new Hamiltonian
            current_H = self.hamiltonian(self.pos)
            self.history.append(current_H)
            
            # Update progress bar
            pbar.set_postfix({'H': f'{current_H:.4f}'})
            
            # Check convergence
            if abs(prev_H - current_H) < self.tolerance:
                break
            prev_H = current_H
    
    def reduce_dimensions(self):
        """Reduce dimensions from medium to final using PCA."""
        from sklearn.decomposition import PCA
        
        # Convert positions to matrix
        nodes = list(self.G.nodes())
        X = np.array([self.pos[node] for node in nodes])
        
        # Apply PCA
        pca = PCA(n_components=self.dim_final)
        X_reduced = pca.fit_transform(X)
        
        # Convert back to dictionary
        return {node: pos for node, pos in zip(nodes, X_reduced)}
    
    def get_pos(self):
        """Get the final 2D positions for visualization."""
        # Optimize in medium dimension
        self.optimize()
        
        # Reduce to final dimension
        final_pos = self.reduce_dimensions()
        
        # Normalize positions to [-1, 1] range
        all_coords = np.array(list(final_pos.values()))
        min_coords = all_coords.min(axis=0)
        max_coords = all_coords.max(axis=0)
        scale = (max_coords - min_coords).max()
        
        if scale > 0:
            normalized_pos = {}
            for node, pos in final_pos.items():
                normalized_pos[node] = 2 * (pos - min_coords) / scale - 1
            return normalized_pos
        
        return final_pos
    
    def plot_convergence(self):
        """Plot the convergence of Hamiltonian during optimization."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history)
        plt.title('Hamiltonian Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Hamiltonian')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Create test graph
    import networkx as nx
    from graph_visualization import GraphVisualizer
    
    # Create a test graph
    G = nx.erdos_renyi_graph(25, 0.5)  # 3x3 grid graph
    
    # Create Hamiltonian layout
    layout = HamiltonianLayout(G, dim_final=2, dim_medium=5, learning_rate=0.05)
    
    # Get positions and visualize convergence
    pos = layout.get_pos()
    layout.plot_convergence()
    
    # Visualize the graph with different layouts
    visualizer = GraphVisualizer()
    
    # Compare layouts
    layouts = {
        'Spring': nx.spring_layout(G),
        'Hamiltonian': pos
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (name, pos) in zip(axes, layouts.items()):
        nx.draw(G, pos=pos, ax=ax, with_labels=True, 
               node_color='lightblue', node_size=500)
        ax.set_title(f'{name} Layout')
    
    plt.tight_layout()
    plt.show()
