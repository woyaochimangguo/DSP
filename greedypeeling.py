import numpy as np
import dsd
import networkx as nx
from dsd import *
from datetime import datetime
import matplotlib.pyplot as plt

#load a small popular graph dataset to illustrate how to run the algorithms
karate_graph = nx.karate_club_graph()
karate_edges = [[e[0],e[1]] for e in nx.karate_club_graph().edges()]
nx.draw(karate_graph)
plt.show()
print('exact max flow method')

start = datetime.now()
exact_R = exact_densest(karate_graph)
print('subgraph induced by', exact_R[0])
print('density =', exact_R[1])
print('run time', datetime.now()-start, '\n')


print('flowless method')
start = datetime.now()
flowless_R = flowless(karate_graph, 5)
print('subgraph induced by', flowless_R[0])
print('density =', flowless_R[1])
print('run time', datetime.now()-start, '\n')

print('greedy method')
start = datetime.now()
greedy_R = greedy_charikar(karate_graph)
print('subgraph induced by', greedy_R[0])
print('density =', greedy_R[1])
print('run time', datetime.now()-start, '\n')


