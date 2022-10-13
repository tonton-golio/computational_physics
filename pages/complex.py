import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
import pandas as pd
import re
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np
import networkx as nx
import sys
sys.setrecursionlimit(15000)

st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)

# -----------
# random walk
def run_random_walk():
    def accumulate(x):
        X=np.zeros(len(x))
        X[0] = x[0]
        
        for i, _ in enumerate(x):
            #st.write(i)
            X[i] = X[i-1]+x[i]
        return X

    def randomWalk(nsteps,
                    sigma2=1, seed=42,
                    xscale='linear', yscale='log'
                    ):
        (dx_f, dy_f) = (lambda theta, r=1: r*trig(theta) for trig in (np.cos, np.sin)) 
        dx_f = lambda theta, r = 1: r*np.cos(theta)
        dy_f = lambda theta, r = 1: r*np.sin(theta)

        np.random.seed(seed)
        thetas_uniform = np.random.uniform(0,2*np.pi,nsteps)
        thetas_randn = np.random.randn(nsteps)*sigma2
        thetas_bimodal = np.concatenate([ np.random.randn(nsteps//2) * sigma2-1, 
                                          np.random.randn(nsteps//2) * sigma2+1 ])

        thetas_uniform[0] = 0
        thetas_randn[0] = 0
        thetas_bimodal[0] = 0

        rands = [thetas_uniform, thetas_randn, thetas_bimodal]
        rands_names = 'uniform, normal, bimodal'.split(', ')

        def plot2():
            colors = 'r g y'.split()
            fig = plt.figure(figsize=(12,6))
            
            gs = GridSpec(3, 3, figure=fig)
            ax1 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
            ax2 = fig.add_subplot(gs[:, 1:])    
            
            lims = {'xl':0, 'xh':0, 'yl':0, 'yh':0}
            for i, (r, n) in enumerate(zip(rands, rands_names)):
                dx, dy = dx_f(r), dy_f(r)
                dx = np.vstack([np.zeros(len(dx)), dx]).T.flatten()
                dy = np.vstack([np.zeros(len(dy)), dy]).T.flatten()
                
                ax1[i].plot(dx,dy, lw=1, c=colors[i])
                ax1[i].set(ylim=(-1,1), xlim=(-1,1), 
                            xticks=[], yticks=[],)
                
                x = accumulate(dx)
                y = accumulate(dy)


                ax2.plot(x,y, lw=2, label=n,c=colors[i])

                ax1[i].set(facecolor = "black", xticks=[], yticks=[], )
                ax2.set(facecolor = "black", xticks=[], yticks=[],xscale=xscale,yscale=yscale)
                ax2.legend(fontsize=20)
                

                
            plt.tight_layout()

            ax1[0].set_title('steps', fontsize=24)
            ax2.set_title('Cummulative', fontsize=24)
            plt.tight_layout()
            st.pyplot(fig)
        plot2()

    nsteps = st.sidebar.slider('nsteps',5,100)
    seed = st.sidebar.slider('seed',0,100)
    sigma2 = st.sidebar.slider('$\simga^2$',0.2,1.)
    xscale = st.sidebar.radio('xscale', ['linear', 'symlog'])
    yscale = st.sidebar.radio('yscale', ['linear', 'symlog'])


    randomWalk(nsteps,sigma2, seed, xscale, yscale)

# -----------
# percolation
def percolation(size,p):
    def makeGrid(size, seed=42): 
        np.random.seed(seed)
        grid = np.random.uniform(0,1,(size,size), )
        grid_with_border = np.ones((size+2,size+2))
        grid_with_border[1:-1, 1:-1] = grid
        return grid_with_border

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
                    pass
        
        new_domains = {}
        index = 0
        for d in domains:
            if len(domains[d]) !=0:
                new_domains[index] = domains[d]
                index += 1
                
        return new_domains

    def run(size, seed):
        grid = makeGrid(size,seed)
        domains = getDomains(grid, p)

        x = np.arange(size+2)
        X,Y = np.meshgrid(x,x)
        
        fig = plt.figure()
        # background
        plt.scatter(X,Y, c='black')

        # colors
        colors = sns.color_palette("hls", len(domains))
        np.random.shuffle(colors)
        colors = np.concatenate([[colors[i]]*len(domains[i]) for i in domains])

        # plot
        xx = np.concatenate([list(domains[i]) for i in domains])
        plt.scatter(xx[:,0], xx[:,1], c=colors)
        plt.xticks([]); plt.yticks([])
        st.pyplot(fig)
    with st.sidebar:
        size = st.slider('size',10,100)
        p = st.slider('$p$',0.01,1.)
        seed = st.slider('seed',10,100)
        run_ = st.radio('run', ['yes', 'no'])

    if run_ == 'yes':
        run(size, seed)

# -----------
# mandel broth
def mandelbroth():
    def stable(z):
        try:
            return False if abs(z) > 2 else True
        except OverflowError:
            return False
    stable = np.vectorize(stable)


    def mandelbroth(c, a, n=10):
        z = 0
        for i in range(n):
            z = z**a + c
        return z

    def make_space(npoints = 5, startRe=-1.85, stopRe=1.85, startIm=-1.85, stopIm=1.85):
        space_im = np.vstack([np.linspace(startIm*1j,stopIm*1j,npoints, dtype=complex)]*npoints).T
        space_re = np.vstack([np.linspace(startRe,stopRe,npoints, dtype=complex)[::-1]]*npoints)
        return space_im+space_re

    def plot_(res):
        fig = plt.figure(figsize=(12,6))
        plt.imshow(res.T, cmap='inferno_r')
        #plt.xticks([]); plt.yticks([])
        plt.xlabel('Im',rotation=0, loc='right')
        plt.ylabel('Re',rotation=0, loc='top')
        st.pyplot(fig)

    with st.sidebar:
        logsize = st.slider('log(size)',1.5,4.)
        size = int(10**logsize)
        st.write(size)
        a = st.slider('a',0.01,13.,2.)
        run_ = st.radio('run', ['yes', 'no'])

    res = stable(mandelbroth(make_space(size), a=a))
    plot_(res)

# -----------
# Bereaucrats
def bereaucrats():
    arr = np.zeros((3,4))
    N = len(arr.flatten())
    nsteps = 100

    _mean = []
    for step in range(nsteps):
        # bring in 1 task
        rand_index = np.random.randint(0,N)
        who = (rand_index//arr.shape[1], rand_index%arr.shape[1])
        arr[who] += 1


        # if someone has 4 tasks redistribute to neighbours
        for i, _ in enumerate(arr):
            for j, _ in enumerate(arr[i]):
                if arr[i,j] >= 4:
                    try: arr[i+1, j] +=1
                    except: pass
                    try: arr[i-1, j] +=1
                    except: pass
                    try: arr[i, j+1] +=1
                    except: pass
                    try: arr[i, j-1] +=1
                    except: pass
                    arr[i,j] -= 4
        _mean.append(np.mean(arr)) 
    _mean = np.array(_mean)

    fig = plt.figure()
    plt.plot(_mean)
    #lt.savefig('bureaucrats.png')
    #plt.close()
    st.pyplot(fig)
    
# -----------
# bakSneppen
def bakSneppen():

    def run(size, nsteps):
        chain = np.random.rand(size)

        X = np.empty((nsteps,size))
        L = np.empty(nsteps)
        for i in range(nsteps):
            lowest = np.argmin(chain)
            chain[(lowest-1+size)%size] = np.random.rand()
            chain[lowest] = np.random.rand()

            chain[(lowest+1)%size] = np.random.rand()
            X[i] = chain
            L[i] = lowest
        

        fig, ax = plt.subplots(2,1)
        ax[0].imshow(X.T, aspect  = nsteps/size*.5, vmin=0, vmax=1)
        ax[1].plot(L)
        st.pyplot(fig)

    with st.sidebar:
        nsteps = st.slider('nsteps',1,30)
        size = st.slider('size',10,31)
        st.write(size)
        run_ = st.radio('run', ['yes', 'no'])

    if run_=="yes":
        run(size, nsteps)

# -----------
# networkGenerator
def network():
    def make_network(n_persons = 5,alpha=.4):
        
        A = np.zeros((n_persons,n_persons))
        for i in range(n_persons):
            neighbours =  np.random.rand(n_persons)>alpha ; neighbours[i]=0
            if sum(neighbours) == 0: 
                a = np.random.randint(0,n_persons,4)
                a = a[a!=i][0]
                neighbours[a] =1
            A[i] += neighbours; A[:,i] = neighbours
        
        return A

    def draw_from_matrix(M, sick=[], pos=[]):
        sick = np.zeros(len(M)) if len(sick) == 0 else sick
        G = nx.Graph()
        for i, line in enumerate(M):
            G.add_node(i)

        for i, line in enumerate(M):
            for j, val in enumerate(line):
                if (i != j) and (val==1): 
                    G.add_edge(i, j)
        color_map = ['r' if s==1 else 'b' for s in sick]
        
        pos = nx.nx_agraph.graphviz_layout(G) if len(pos)==0 else pos
        
        nx.draw_networkx(G,pos, node_color=color_map)
        return pos


    with st.sidebar:
        N = st.slider('N',1,42,22)
        a = st.slider('alpha', 0.,1.,0.97)
    fig, ax = plt.subplots()

    net = make_network(N,a)
    draw_from_matrix(net)
    st.pyplot(fig)

viz = st.sidebar.selectbox('viz', ['RandomWalk 2d', 
                                    'Percolation', 'mandelbroth',
                                    'bereaucrats', 'bakSneppen',
                                    'network'])
if viz =='RandomWalk 2d':
    run_random_walk()
elif viz =='Percolation':
    percolation(size=20,p=.2)
elif viz =='mandelbroth':
    mandelbroth()
elif viz =='bereaucrats':
    bereaucrats()
elif viz =='bakSneppen':
    bakSneppen()
elif viz == 'network':
    network()







#plt.style.available











