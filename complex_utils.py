import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

try: import graphviz # having trouble with this when hosted
except: pass
try: import networkx as nx # having trouble with this when hosted
except: pass

# gerneral
def escapeCharacters(text, inverted=True):
    ecs = {'\0' : '\x00',   # null
            '\t' : '\x09',  # tab
            '\r' : '\x0d',  # carriage return
            '\n' : '\x0a',
            r'\f' : r'\x0c',
        }
    
    for c in ecs:
        if inverted: text = text.replace(ecs[c], c)
        else:        text = text.replace(c, ecs[c])

    return text

def getText_prep(filename = 'pages/stat_mech.md', split_level = 2):
    with open(filename,'r' ) as f:
        file = escapeCharacters(f.read())
    level_topics = file.split('\n'+"#"*split_level+' ')
    text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
    return text_dict    


# statatistical mechanics
def ising(size, nsteps, beta, nsnapshots):
    # initialize
    X = np.random.rand(size,size)
    X[X>0.5] =1 ; X[X!=1] =-1
    E = 0 
    for i in range(size):
        for j in range(size):
            sum_neighbors = 0
            for pos in [(i,(j+1)%size),    (i,(j-1+size)%size), 
                            ((i+1)%size,j), ((i-1+size)%size,j)]:
                
                sum_neighbors += X[pos]
        E += -X[i,j] * sum_neighbors/2

    results = {"Energy" : [E], 
                "Magnetization" : [np.sum(X)], 
                "snapshots": {} }
    for step in range(nsteps):
        (i,j) = tuple(np.random.randint(0,size-1,2)) #choose random site

        sum_neighbors = 0
        for pos in [(i,(j+1)%size),    (i,(j-1+size)%size), 
                        ((i+1)%size,j), ((i-1+size)%size,j)]:
            sum_neighbors += X[pos]
            
        dE = 2 *X[i,j] * sum_neighbors

        
        if np.random.rand()<np.exp(-beta*dE):
            X[i,j]*=-1
            E += dE

        results['Energy'].append(E.copy())
        results['Magnetization'].append(np.sum(X))
        if step in np.arange(nsnapshots)*nsteps//nsnapshots:
            results['snapshots'][step]=X.copy()

    # load, fill and save susceptibility data
    try: data = np.load('pages/data.npz', allow_pickle=True)[np.load('pages/data.npz', allow_pickle=True).files[0]].item()
    except: data = {};  np.savez('pages/data', data)

    susceptibility = np.var(results['Magnetization'][-nsteps//4*3:])
    data[beta] = {'sus': susceptibility, 'nsteps':nsteps, 'size':size}
    np.savez('pages/data', data)
    return results, data

def plotSnapshots(results, nsnapshots):
    
    fig, ax = plt.subplots(3,nsnapshots //3, figsize=(9,9))
    for idx, key in enumerate(results['snapshots'].keys()):
        ax[idx//3, idx%3].imshow(results['snapshots'][key])
        ax[idx//3, idx%3].set(xticks=[], yticks=[]) 
        ax[idx//3, idx%3].set_title(key, color="white")
        plt.tight_layout()
    return fig

def plotEnergy_magnetization(results):
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    ax2 = ax.twinx()
    ax.plot(results['Energy'],c='red', lw=2)

    ax2.plot(np.abs(results['Magnetization']), color='orange')
    
    ax.set_xlabel('Timestep', color='white')
    ax.set_ylabel('Energy', color='red')
    ax2.set_ylabel('|Magnetization|', color='orange')
    
    return fig

def plotSusceptibility(data):
    ## susceptibility plot

    fig, ax = plt.subplots( figsize=(5,3))
    ax.scatter(x = list(data.keys()), 
                    y = [data[key]['sus'] for key in data.keys()],
                    s = [data[key]['size'] for key in data.keys()],
                    color='cyan')
    ax.set_ylabel('Susceptibility', color='white')
    ax.set_xlabel('beta', color='white')

    return fig

def metropolisVisualization(beta):
    dEs = np.linspace(-1,3,1000)
    prob_change = np.exp(-beta*dEs)
    prob_change[dEs<0] = 1

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(dEs, prob_change, color='pink', lw=7)
    ax.set_ylabel('probability of acceptance', color='white')
    ax.set_xlabel('Energy difference', color='white')
    ax.set(xticks=[0])
    plt.grid()

    return fig


# Percolation and Fractals


def percolation(size, seed, p,marker):
    def makeGrid(size, seed=42): 
        np.random.seed(seed)
        grid = np.ones((size+2,size+2))
        grid[1:-1, 1:-1] = np.random.uniform(0,1,(size,size))
        return grid

    def checkNeighbours(pos, grid, domain, visited):
        (i,j) = pos
        neighbours = [(i-1,j), (i+1,j), (i,j-1), (i, j+1)]
        for n in neighbours:
            if (n[0]>=0) and (n[1]>=0) and (n[0]<len(grid)) and (n[1]<len(grid)):
                if grid[n] and (n not in visited):
                    domain.add(n)
                    visited.add(n)
                    domain_, visited_ = checkNeighbours(n, grid, domain, visited)
                    domain = domain.union(domain_)
                    visited = visited.union(visited_)
                else: visited.add(n)
        return domain, visited

    def getDomains(grid, p=.5):
        open_arr = grid < p
        domains = {} ; index = 0; visited = set()
        for i, _ in enumerate(open_arr):
            for j, val in enumerate(open_arr[i]):
                if val:
                    if (i,j) in visited:
                        domain, visited_ = checkNeighbours((i,j), open_arr, domain=set(), visited=visited)
                    else:
                        visited.add((i,j))
                        domain, visited_ = checkNeighbours((i,j), open_arr, domain=set([(i,j)]), visited=visited)
                    domains[index] = domain
                    visited = visited.union(visited_)
                    index+=1
                else:
                    visited.add((i,j))
        
        new_domains = {}
        index = 0
        for d in domains:
            if len(domains[d]) !=0:
                new_domains[index] = domains[d]
                index += 1
                
        return new_domains

    grid = makeGrid(size,seed)
    domains = getDomains(grid, p)

    x = np.arange(size+2)
    X,Y = np.meshgrid(x,x)
    
    fig, ax = plt.subplots()
    # background
    ax.scatter(X,Y, c='black')

    # colors
    colors = sns.color_palette("hls", len(domains))
    np.random.shuffle(colors)
    colors = np.concatenate([[colors[i]]*len(domains[i]) for i in domains])

    # plot
    xx = np.concatenate([list(domains[i]) for i in domains])
    ax.scatter(xx[:,0], xx[:,1], c=colors, marker=marker)
    ax.set(xticks = [], yticks = [], facecolor='black')
    return fig, domains

def betheLattice_old():
    # Create a graphlib graph object
    graph = graphviz.Digraph()

    root = str(0)
    nodes = []
    for other in '0 1 2'.split():
        graph.edge(root, root+other)
        nodes.append(root+other)

    new_nodes = []
    for i in nodes:
        for j in range(2):
            graph.edge(str(i), str(i)+str(j))

            new_nodes.append(str(i)+str(j))

    nodes = new_nodes
    new_nodes = []
    for i in nodes:
        for j in range(2):
            graph.edge(str(i), str(i)+str(j))
            new_nodes.append(str(i)+str(j))
    return graph


def betheLattice(p=0.1, size=62, get_many=False, ps=[.5]):
    def makeBetheLattice(n_nodes = 10):
        M = np.zeros((n_nodes,n_nodes))

        idx = 1
        for i, _ in enumerate(M):
            if i ==0: n =3
            else: n = 2
            M[i, idx:idx+n] = 1
            idx+=n
        return M+M.T

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
            for i in d: inDomain[i] = idx   
        
        G = nx.Graph()
        for i, line in enumerate(M): G.add_node(i)

        for i, line in enumerate(M):
            for j, val in enumerate(line):
                if (i != j) and (val==1): 
                    G.add_edge(i, j)
        palette = sns.color_palette('hls', len(domains))
        color_map = ['darkgrey' if i not in inDomain.keys() else palette[inDomain[i]] for i in range(len(M))]

        if len(palette) == 0: color_map = ['orange'] * len(M)

        fig = plt.figure()
        nx.draw_networkx(G, node_color=color_map, pos=nx.kamada_kawai_layout(G))
        return fig
        
    if get_many == False:
        M = makeBetheLattice(size)
        domains = getDomains(M,p)
        return draw_from_matrix(M,domains)

    else:
        Ns = {}
        M = makeBetheLattice(size)
        for p in ps:
            Ns[p] = len(getDomains(M,p))
        return Ns


def run_fractals(size_fractal, a ,n):

    def stable(z):
        try:
            return False if abs(z) > 2 else True
        except OverflowError:
            return False
    stable = np.vectorize(stable)


    def mandelbrot(c, a, n=50):
        z = 0
        for i in range(n):
            z = z**a + c
        return z

    def makeGrid(resolution, lims=[-1.85, 1.25, -1.25, 1.45]):
        re = np.linspace(lims[0], lims[1], resolution)[::-1]
        im = np.linspace(lims[2], lims[3], resolution)
        re, im = np.meshgrid(re,im)
        return re+im*1j

    def plot_(res):
        fig = plt.figure(figsize=(12,6))
        plt.imshow(res.T, cmap='magma')
        plt.xticks([]); plt.yticks([])
        plt.xlabel('Im',rotation=0, loc='right', color='blue')
        plt.ylabel('Re',rotation=0, loc='top', color='blue')
        return fig

    res = stable(mandelbrot(makeGrid(size_fractal,  lims=[-1.85, 1.25, -1.25, 1.45]), a=a, n=n))
    return plot_(res)


