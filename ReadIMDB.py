#!/usr/bin/env python3
import csv
import numpy as np
from scipy.sparse import lil_matrix

# 手动设置输入和输出文件路径
IMDBFile = "C:\\Users\\ECNU\\Documents\\GitHub\\DSP\\datasets\\IMDB\\IMDB.mtx"
EdgeFile = "C:\\Users\\ECNU\\Documents\\GitHub\\DSP\\datasets\\IMDB\\edges.csv"
DegFile = "C:\\Users\\ECNU\\Documents\\GitHub\\DSP\\datasets\\IMDB\\deg.csv"


############################# Read the IMDB file ##############################
# [output1]: movie_actor_lst ([movie_id, actor_id])
# [output2]: movie_num
# [output3]: actor_num
def ReadIMDB():
    # Initialization
    movie_actor_lst = []

    # Read the IMDB file
    f = open(IMDBFile, "r")
    for i, line in enumerate(f):
        # Skip the header
        if i < 55:
            continue
        # Read #movies and #actors --> movie_num, actor_num
        elif i == 55:
            lst = line.rstrip("\n").split(" ")
            movie_num = int(lst[0])
            actor_num = int(lst[1])
        # Read the movie-actor list --> movie_actor_lst
        else:
            lst = line.rstrip("\n").split(" ")
            movie_id = int(lst[0])
            actor_id = int(lst[1])
            movie_actor_lst.append([movie_id, actor_id])
    f.close()

    return movie_actor_lst, movie_num, actor_num

#################################### Main #####################################
# Read the IMDB file
movie_actor_lst, movie_num, actor_num = ReadIMDB()

# Make a movie dictionary ({movie_id: [actor_id]}) --> movie_dic
# (Both movie_id and actor_id start with zero)
movie_dic = {}
for i in range(movie_num):
    movie_dic[i] = []
for lst in movie_actor_lst:
    # Both movie_id and actor_id start with zero
    movie_id = lst[0] - 1
    actor_id = lst[1] - 1
    movie_dic[movie_id].append(actor_id)

# Make edges --> edges_lil
print("Making edges.")
edges_lil = lil_matrix((actor_num, actor_num))
deg = np.zeros(actor_num)
for i in range(movie_num):
    if i % 10000 == 0:
        print(i)
    for j in range(len(movie_dic[i])):
        for k in range(j + 1, len(movie_dic[i])):
            # actor indexes --> actor1, actor2
            actor1 = movie_dic[i][j]
            actor2 = movie_dic[i][k]
            if edges_lil[actor1, actor2] == 0:
                edges_lil[actor1, actor2] = 1
                deg[actor1] += 1
                deg[actor2] += 1
a1, a2 = edges_lil.nonzero()
print("#edges:", len(a1))

# Output edge information
print("Outputting edge information.")
with open(EdgeFile, "w", newline='') as f:
    print("#nodes", file=f)
    print(actor_num, file=f)
    print("node,node", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for i in range(len(a1)):
        # actor indexes --> lst
        lst = [a1[i], a2[i]]
        writer.writerow(lst)

# Output degree information
print("Outputting degree information.")
with open(DegFile, "w", newline='') as f:
    print("node,deg", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for actor1 in range(actor_num):
        # actor index and her degree --> lst
        lst = [actor1, int(deg[actor1])]
        writer.writerow(lst)


import networkx as nx
# 创建无向图对象
G = nx.Graph()
# 读取文件并添加边
with open("C:\\Users\\ECNU\\Documents\\GitHub\\DSP\\datasets\\IMDB\\edges.csv", "r") as f:
    for line in f:
        # 去掉行尾换行符并按逗号分割每一行
        node1, node2 = map(int, line.strip().split(","))
        # 添加边到图中
        G.add_edge(node1, node2)
# 图构建完成
print("Graph has been created.")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())