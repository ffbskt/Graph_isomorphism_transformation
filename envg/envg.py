from visg import VisG
import networkx as nx


# create a graph
G = nx.Graph()

# add nodes
G.add_node('s', type='singleton', label='s')
G.add_node('c_0', type='reward', label='c0')
G.add_node('c_1', type='reward', label='c1')




vis = VisG()
vis.add_graph(G)
vis.draw(layout='spring', title='Box Pattern - Hamiltonian Layout')
vis.show()
