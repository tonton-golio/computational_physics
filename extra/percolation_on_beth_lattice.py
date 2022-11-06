import matplotlib.pyplot as plt
import numpy as np

def makeBetheLattice(n_nodes = 10):
    M = np.zeros((n_nodes,n_nodes))

    idx = 1
    for i, _ in enumerate(M):
        if i ==0: n =3
        else: n =2
        M[i, idx:idx+n] = 1
        idx+=n

    return M+M.T

import seaborn as sns

def checkOpenNeighbours(open_neighbours,visited, domain, open_arr):
    
    for j in open_neighbours:
        if j not in visited:
            domain.append(j)
            visited.add(j)
            open_neighbours = np.argwhere(M[j] * open_arr).flatten()
            open_neighbours = open_neighbours[open_neighbours!=j]
            visited_, domain_ = checkOpenNeighbours(open_neighbours,visited, domain,open_arr)
            visited = visited.union(visited_)
            domain += domain
    return visited, domain
    
def getDomains(M, p=0.3):
    
    open_arr = p>np.random.rand(len(M))
    visited = set()
    domains = []
    for i in range(len(M)):

        if i not in visited:
            if open_arr[i]:
                domain = []
                visited.add(i)
                domain.append(i)

                open_neighbours = np.argwhere(M[i] * open_arr).flatten()
                open_neighbours = open_neighbours[open_neighbours!=i]
                if len(open_neighbours)>0:
                    visited_, domain_ = checkOpenNeighbours(open_neighbours,visited, domain,open_arr)
                    visited = visited.union(visited_)
                    domain += domain
                domains.append(set(domain))

    return domains
                
            


def draw_from_matrix(M, domains) :
    
    inDomain = {}
    for idx, d in enumerate(domains):
        for i in d:
            inDomain[i] = idx
    inDomain   
    
    
    G = nx.Graph()
    for i, line in enumerate(M):
        G.add_node(i)

    for i, line in enumerate(M):
        for j, val in enumerate(line):
            if (i != j) and (val==1): 
                G.add_edge(i, j)
    palette = sns.color_palette('hls', len(domains))
    color_map = ['darkgrey' if i not in inDomain.keys() else palette[inDomain[i]] for i in range(len(M))]

    nx.draw_networkx(G, node_color=color_map, pos=nx.kamada_kawai_layout(G))
    
    
import networkx as nx
import numpy as np
M = makeBetheLattice(60)
domains = getDomains(M,0.6)
open_arr = draw_from_matrix(M,domains)