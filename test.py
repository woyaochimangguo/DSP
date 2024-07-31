from exdata import *
import numpy as np
import dsd
import networkx as nx
from dsd import *
from datetime import datetime
import matplotlib.pyplot as plt

G = nx.read_edgelist('./datasets/Facebook/facebook/0.edges', nodetype=int)
circles = read_circles("./datasets/Facebook/facebook/0.circles")
#lab = explore_and_summarize_network(edgelist = G.edges(), vertices=G.nodes(), subgraph=None, circles=circles)
print("node is ",G.nodes)
print("edge is ",G.edges)
karate_graph = G
#karate_edges = [[e[0],e[1]] for e in nx.karate_club_graph().edges()]
nx.draw(karate_graph)
plt.show()
print('exact max flow method')

start = datetime.now()
exact_R = exact_densest(karate_graph)
print('subgraph induced by', exact_R[0])
print('density =', exact_R[1])
print('run time', datetime.now()-start, '\n')


