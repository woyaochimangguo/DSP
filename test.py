from exdata import *
import numpy as np
import dsd
import networkx as nx
from dsd import *
from datetime import datetime
import matplotlib.pyplot as plt
from greedypeeling import charikar_peeling
from spectral import *
G = nx.read_edgelist('./datasets/Facebook/facebook/414.edges', nodetype=int)
# circles = read_circles("./datasets/Facebook/facebook/414.circles")
# lab = explore_and_summarize_network(edgelist = G.edges(), vertices=G.nodes(), subgraph=None, circles=circles)
print("node is ",G.nodes)
# print("edge is ",G.edges)
print("the original graph density is",len(G.edges) / len(G.nodes))
karate_graph = G
print("G的边",G.edges)
#karate_edges = [[e[0],e[1]] for e in nx.karate_club_graph().edges()]
# nx.draw(karate_graph)
# plt.show()
# print('exact max flow method')
# #
# start = datetime.now()
# exact_R = exact_densest(karate_graph)
# print('subgraph induced by', exact_R[0])
# print('density =', exact_R[1])
# print('run time', datetime.now()-start, '\n')
#
#
# print('flowless method')
# start = datetime.now()
# flowless_R = flowless(karate_graph, 5)
# print('subgraph induced by', flowless_R[0])
# print('density =', flowless_R[1])
# print('run time', datetime.now()-start, '\n')
#
# print('greedy method')
# start = datetime.now()
# greedy_R = greedy_charikar(karate_graph)
# print('subgraph induced by', greedy_R[0])
# print('density =', greedy_R[1])
# print('run time', datetime.now()-start, '\n')

#
# print("greedypelling method")
# start = datetime.now()
# dense_subgraph,density = charikar_peeling(G)
# # Print the nodes and edges of the dense subgraph
# print('run time', datetime.now()-start, '\n')
# print("Nodes in dense subgraph:", dense_subgraph.nodes())
# print("Density of the dense subgraph:", density)
# print("Edges in dense subgraph:", dense_subgraph.edges())

# print("Spectral method")
# start = datetime.now()
# densest_subgraph_sp, density_sp = general_sweep_algorithm(G)
# print('run time', datetime.now()-start, '\n')
# print("Nodes in the densest subgraph:", densest_subgraph_sp)
# print("Density of the densest subgraph:", density_sp)
#
# GG = nx.gnm_random_graph(n=50, m=200)
print("Spectral_lp method")
start = datetime.now()
num_edges,density_ = general_sweep_algorithm_laplacian_final(G)
# densest_subgraph_sp, density_sp = general_sweep_algorithm_laplacian_1(G)
print('run time', datetime.now()-start, '\n')
# print("Nodes in the densest subgraph:", densest_subgraph_sp)
# print("Density of the densest subgraph:", density_sp)