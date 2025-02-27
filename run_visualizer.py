import numpy as np
import matplotlib.pyplot as plt
from graph_from_image import img2graph, add_special_nodes
from graph_visualization import GraphVisualizer

# Create a 5x5 image with 1s and 0s
img = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
], dtype=np.uint8)

# Convert image to graph
G = img2graph(img)

# Add special nodes (start and boundary)
#G = add_special_nodes(G)

# Create visualizer
visualizer = GraphVisualizer()

# Create a figure with two subplots for different layouts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Draw with grid layout
visualizer.draw_graph(G, layout='grid_square', title="Image Graph with Grid Layout", ax=ax1)

# Draw with Hamiltonian layout
visualizer.draw_graph(G, layout='hamiltonian', title="Image Graph with Hamiltonian Layout", ax=ax2)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the visualization
plt.savefig('image_graph_visualization.png', bbox_inches='tight', dpi=300) 