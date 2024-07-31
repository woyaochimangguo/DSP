import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
from collections import Counter
import random
from itertools import combinations, groupby
import networkx as nx
import numpy as np
from networkx.algorithms import approximation
from networkx.algorithms import reciprocity
import matplotlib.pyplot as plt
from collections import Counter
import random
from itertools import combinations, groupby
from node2vec import Node2Vec
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
def explore_and_summarize_network(edgelist, vertices, subgraph, directed=False):
    """
    Concisely summarizes any induced subgraph of the input network
    """
    G = nx.Graph()
    if subgraph is not None:
        induced_edges = [edge for edge in edgelist if ((edge[0] in subgraph) and (edge[1] in subgraph))]
        G.add_nodes_from(subgraph)
        G.add_edges_from(induced_edges, nodetype=int)
    else:
        G.add_nodes_from(vertices)
        G.add_edges_from(edgelist, nodetype=int)

    # (a)
    nx.draw(G, pos=nx.spring_layout(G), node_color='maroon',
            node_size=20,
            edge_color="gray",
            width=0.5)
    plt.show()
    # (b)
    if directed:
        in_degree_sequence = [item[-2] for item in local_summaries(G)]
        in_degree_counts = Counter(degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(in_degree_counts.keys(), in_degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('In Degrees')
        ax.set_title(r'Histogram of in_degrees')
        fig.tight_layout()
        plt.show()

        out_degree_sequence = [item[-1] for item in local_summaries(G)]
        out_degree_counts = Counter(out_degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(out_degree_counts.keys(), out_degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('out Degrees')
        ax.set_title(r'Histogram of out_degrees')
        fig.tight_layout()
        plt.show()
    else:
        degree_sequence = [item[-1] for item in local_summaries(G)]
        degree_counts = Counter(degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(degree_counts.keys(), degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Degrees')
        ax.set_title(r'Histogram of degrees')
        fig.tight_layout()
        plt.show()
        # (c)
    betweenness_centrality = [item[0] for item in local_summaries(G)]
    fig1, ax1 = plt.subplots()
    ax1.hist(betweenness_centrality)
    ax1.set_xlabel('Nodes')
    ax1.set_ylabel('Betweenness Centrality')
    ax1.set_title(r'Histogram of Betweenness Centrality')
    fig1.tight_layout()
    plt.show()
    # (d)
    eigenvector_centrality = [item[1] for item in local_summaries(G)]
    fig2, ax2 = plt.subplots()
    ax2.hist(eigenvector_centrality)
    ax2.set_xlabel('Nodes')
    ax2.set_ylabel('Eigenvector Centrality')
    ax2.set_title(r'Histogram of Eigenvector Centrality')
    fig2.tight_layout()
    plt.show()
    # Print Global Summaries
    global_summaries(G)

    def local_summaries(G, directed=False):
        betweenness_centrality = nx.centrality.betweenness_centrality(G)
        eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
        closeness_centrality = nx.centrality.closeness_centrality(G)
        if directed:
            in_degrees = [G.in_degree(n) for n in G.nodes]
            out_degrees = [G.in_degree(n) for n in G.nodes]
            return zip(betweenness_centrality.values(), eigenvector_centrality.values(), closeness_centrality.values(),
                       in_degrees, out_degrees)
        else:
            degrees = [G.degree(n) for n in G.nodes]
            return zip(betweenness_centrality.values(), eigenvector_centrality.values(), closeness_centrality.values(),
                       degrees)

def global_summaries(G):
    try:
        diameter = nx.algorithms.distance_measures.diameter(G)
    except:
        diameter = "Found infinite path length because the graph is not connected !"
    clustering_coefficient = nx.algorithms.approximation.clustering_coefficient.average_clustering(G)
    number_of_nodes = G.number_of_edges()
    number_of_edges = G.number_of_nodes()
    number_of_connected_components = nx.number_connected_components(G)
    largest_connected_component = max([ len(i) for i in list(nx.connected_components(G))])
    print("##### Global Summaries #####")
    print("Diameter : ",diameter)
    print("Number of Nodes : ",number_of_nodes)
    print("Number of Edges : ",number_of_edges)
    print("Number of Connected Components : ",number_of_connected_components)
    print("Size of the Largest Connected Compopnent : ",largest_connected_component)

def read_circles(file_path):
    with open(file_path) as f :
        content = f.readlines()
        content = [line.replace("\n","") for line in content]
        content = [line.split("\t") for line in content]
    return {circle[0]:list(map(int,circle[1:])) for circle in content} # Convert nodes to int and add to dict.

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G
def node2vec_from_graph(G, D):
    node2vec = Node2Vec(G, dimensions=D)
    model = node2vec.fit()
    embeddings = model.wv.vectors
    return embeddings
def get_circle_labels(G,circles):
    circle_labels = []
    for node in G.nodes():
        is_added = False
        for i,li in enumerate(list(circles.values())):
            if node in li and not is_added:
                circle_labels.append(i)
                is_added = True
        if not is_added:
            circle_labels.append(-1)
    return circle_labels
def get_community_labels(G, circles):
    community_labels = []
    for node in G.nodes():
        is_added = False
        for i,li in enumerate(lab):
            if node in li and not is_added:
                community_labels.append(i)
                is_added = True
        if not is_added:
            community_labels.append(-1)
    return community_labels


def colorize_nodes(labels, G):
    final = [(0, 0, 0) for i in range(len(G.nodes()))]
    colors = [(np.random.random(1)[0], np.random.random(1)[0], np.random.random(1)[0]) for i in range(len(labels))]

    for j, n in enumerate(G.nodes()):
        for i, label in enumerate(labels):
            if n in label:
                final[j] = colors[i]
    return final


def explore_and_summarize_network(edgelist, vertices, subgraph, directed=False, circles=None):
    """
    Concisely summarizes any induced subgraph of the input network
    """
    G = nx.Graph()
    if subgraph is not None:
        induced_edges = [edge for edge in edgelist if ((edge[0] in subgraph) and (edge[1] in subgraph))]
        G.add_nodes_from(subgraph)
        G.add_edges_from(induced_edges, nodetype=int)
    else:
        G.add_nodes_from(vertices)
        G.add_edges_from(edgelist, nodetype=int)
    G = list(G.subgraph(c) for c in nx.connected_components(G))[0]
    sum_list, labels, modularity, isolates = local_summaries_v2(G)
    # (a)
    nx.draw(G, pos=nx.fruchterman_reingold_layout(G), node_color=colorize_nodes(labels, G),
            node_size=20,
            edge_color="gray",
            width=0.5, cmap=plt.cm.Blues)
    plt.show()
    if circles:
        nx.draw(G, pos=nx.fruchterman_reingold_layout(G),
                node_color=colorize_nodes([set(item) for item in circles.values()], G),
                node_size=20,
                edge_color="gray",
                width=0.5, cmap=plt.cm.Blues)

    # (b)
    if directed:
        in_degree_sequence = [item[-2] for item in local_summaries_v2(G)[0]]
        in_degree_counts = Counter(degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(in_degree_counts.keys(), in_degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('In Degrees')
        ax.set_title(r'Histogram of in_degrees')
        fig.tight_layout()
        plt.show()

        out_degree_sequence = [item[-1] for item in local_summaries_v2(G)[0]]
        out_degree_counts = Counter(out_degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(out_degree_counts.keys(), out_degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('out Degrees')
        ax.set_title(r'Histogram of out_degrees')
        fig.tight_layout()
        plt.show()

        reciprocity_sequence = [item[-1] for item in local_summaries_v2(G)[0]]
        reciprocity_counts = Counter(reciprocity_sequence)
        fig, ax = plt.subplots()
        ax.bar(reciprocity_counts.keys(), reciprocity_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('out Degrees')
        ax.set_title(r'Histogram of Reciprocity')
        fig.tight_layout()
        plt.show()
    else:
        degree_sequence = [item[-1] for item in local_summaries_v2(G)[0]]
        degree_counts = Counter(degree_sequence)
        fig, ax = plt.subplots()
        ax.bar(degree_counts.keys(), degree_counts.values())
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Degrees')
        ax.set_title(r'Histogram of degrees')
        fig.tight_layout()
        plt.show()
        # (c)
    betweenness_centrality = [item[0] for item in local_summaries_v2(G)[0]]
    fig1, ax1 = plt.subplots()
    ax1.hist(betweenness_centrality)
    ax1.set_xlabel('Nodes')
    ax1.set_ylabel('Betweenness Centrality')
    ax1.set_title(r'Histogram of Betweenness Centrality')
    fig1.tight_layout()
    plt.show()
    # (d)
    eigenvector_centrality = [item[1] for item in local_summaries_v2(G)[0]]
    fig2, ax2 = plt.subplots()
    ax2.hist(eigenvector_centrality)
    ax2.set_xlabel('Nodes')
    ax2.set_ylabel('Eigenvector Centrality')
    ax2.set_title(r'Histogram of Eigenvector Centrality')
    fig2.tight_layout()
    plt.show()
    # Print Global Summaries
    global_summaries(G)
    print("Modularity : ", modularity)
    return labels


def local_summaries_v2(G, directed=False):
    betweenness_centrality = nx.centrality.betweenness_centrality(G)
    max_betweenness_centrality_node = max(betweenness_centrality, key=betweenness_centrality.get)
    eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
    max_eigenvector_centrality = max(eigenvector_centrality, key=eigenvector_centrality.get)
    closeness_centrality = nx.centrality.closeness_centrality(G)
    max_closeness_centrality = max(closeness_centrality, key=closeness_centrality.get)

    if directed:
        in_degrees = [G.in_degree(n) for n in G.nodes]
        in_degree_dict = {n: G.degree(n) for n in G.nodes}
        max_in_degree = max(max_in_degree, key=max_in_degree.get)
        out_degrees = [G.in_degree(n) for n in G.nodes]
        out_degree_dict = {n: G.degree(n) for n in G.nodes}
        max_out_degree = max(out_degree_dict, key=out_degree_dict.get)
        reciprocity = nx.algorithms.reciprocity(G)
        mean_reciprocity = [value for key, value in nx.algorithms.reciprocity(G)].mean()
        return zip(betweenness_centrality.values(),
                   eigenvector_centrality.values(),
                   closeness_centrality.values(),
                   in_degrees,
                   out_degrees,
                   reciprocity,
                   nx.clustering(
                       G)), max_betweenness_centrality_node, max_eigenvector_centrality, max_closeness_centrality, max_in_degree, max_out_degree, mean_reciprocity, nx.algorithms.isolate.number_of_isolates(
            G)
    else:
        degrees = [G.degree(n) for n in G.nodes]
        partitions = list(nx.algorithms.community.centrality.girvan_newman(G))
        mod_scores = [nx.community.quality.modularity(G, partition) for partition in partitions]
        maximum_modularity_score = max(mod_scores)
        best_partition = partitions[mod_scores.index(maximum_modularity_score)]
        return zip(betweenness_centrality.values(),
                   eigenvector_centrality.values(),
                   closeness_centrality.values(),
                   degrees,
                   nx.clustering(
                       G)), best_partition, maximum_modularity_score, nx.algorithms.isolate.number_of_isolates(G)

def global_summaries(G):
    try:
        diameter = nx.algorithms.distance_measures.diameter(G)
    except:
        diameter = "Found infinite path length because the graph is not connected !"
    clustering_coefficient = nx.algorithms.approximation.clustering_coefficient.average_clustering(G)
    number_of_nodes = G.number_of_edges()
    number_of_edges = G.number_of_nodes()
    number_of_connected_components = nx.number_connected_components(G)
    largest_connected_component = max([ len(i) for i in list(nx.connected_components(G))])
    print("##### Global Summaries #####")
    print("Diameter : ",diameter)
    print("Number of Nodes : ",number_of_nodes)
    print("Number of Edges : ",number_of_edges)
    print("Number of Connected Components : ",number_of_connected_components)
    print("Size of the Largest Connected Compopnent : ",largest_connected_component)

def read_circles(file_path):
    with open(file_path) as f :
        content = f.readlines()
        content = [line.replace("\n","") for line in content]
        content = [line.split("\t") for line in content]
    return {circle[0]:list(map(int,circle[1:])) for circle in content} # Convert nodes to int and add to dict.