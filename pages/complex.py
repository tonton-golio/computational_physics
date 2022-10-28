import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
try: import graphviz # having trouble with this when hosted
except: pass
try: import networkx as nx  # networkx too :(
except: pass
import time
import sys
sys.setrecursionlimit(15000)

st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)

# matplotlib style

mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['axes.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# ax.set_ylabel('probability of acceptance', color='white')
# ax.set_xlabel('Energy difference', color='white')
# ax.set(xticks=[0])
# plt.grid()
# fig.patch.set_facecolor((.04,.065,.03))
# plt.tight_layout()


def run_stat_mech():
    # Sidebar
    with st.sidebar:
        size = st.slider('size',3,100,10)
        beta = st.slider('beta',0.01,5.,1.)
        nsteps = st.slider('nsteps',3,10000,100)
        nsnapshots = 4

    # functions
    def metropolisVisualization():
        dEs = np.linspace(-1,3,1000)
        prob_change = np.exp(-beta*dEs)
        prob_change[dEs<0] = 1

        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(dEs, prob_change, color='pink', lw=7)

        ax.set(facecolor=(.04,.065,.03))
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_ylabel('probability of acceptance', color='white')
        ax.set_xlabel('Energy difference', color='white')
        ax.set(xticks=[0])
        plt.grid()
        # fig.patch.set_facecolor((.04,.065,.03))
        plt.tight_layout()

        return fig

    def ising():
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

    def plotSnapshots(nsnapshots = 4):
        fig, ax = plt.subplots(1,nsnapshots, figsize=(15,3))
        for idx, key in enumerate(results['snapshots'].keys()):
            ax[idx].imshow(results['snapshots'][key])
        # fig.patch.set_facecolor((.04,.065,.03))
        return fig

    def plotEnergy_magnetization():
        fig, ax = plt.subplots(2,1, figsize=(5,6))
        ax[0].plot(results['Energy'],c='purple')
        ax[1].plot(results['Magnetization'], color='orange')
        

        for i in [0,1]: # could we make plotstyle page-wide?
            ax[i].set(facecolor=(.04,.065,.03))
            ax[i].tick_params(axis='x', colors='white')
            ax[i].tick_params(axis='y', colors='white')
            ax[i].set_xlabel('Timestep', color='white')
        
        ax[0].set_ylabel('Energy', color='white')
        ax[1].set_ylabel('Magnetization', color='white')
        
        # fig.patch.set_facecolor((.04,.065,.03))
        plt.tight_layout()
        return fig
    
    def plotSusceptibility():
        ## susceptibility plot

        fig, ax = plt.subplots( figsize=(5,3))
        ax.scatter(x = list(data.keys()), 
                      y = [data[key]['sus'] for key in data.keys()],
                      s = [data[key]['size'] for key in data.keys()],
                      color='cyan')
        ax.set_ylabel('Susceptibility', color='white')
        ax.set_xlabel('beta', color='white')

        ax.set(facecolor=(.04,.065,.03))
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # fig.patch.set_facecolor((.04,.065,.03))
        return fig

    # Render

    st.markdown(r"""
        # Statistical Mechanics
        
        ## Partition function
        
        ### Microcanonical Ensemble
        The central assumption of statistical mechanics is "principle of 
        equal a priori probabilities" which argues that all (quantum) states 
        with same energy $E$ of the closed mactoscopic system exists equally 
        likely.
        With this assumption one can say that the system become particular state
        $i$ with probability
        $$
            P_i = \frac{1}{\Omega}
        $$
        Here $\Omega$ is the total number of (quantum) microstate of the 
        system with energy $E$.


        According to Ludwig Boltzmann, entropy of system with microcanonical 
        ensemble of the system with fixed energy $E$ is expressed as
        $$
            S = k_\mathrm{B} \ln \Omega.
        $$ 
        Here $k_\mathrm{B}$ is Boltzmann's constant.
        
        In this system, temperature $T$ is statistically defined as 
        $$ 
            \frac{1}{T} = \frac{\partial S}{\partial E} 
            = k_\mathrm{B} \frac{\partial \ln \Omega}{\partial E} 
        $$ 

        ### Canonical Ensemble
        Let's consider subsystem of whole system of microcanonical ensemble. 
        For simplicity, we only consider only one subsystem and assuming it 
        is small enough comparing to the rest of the system. Let's say this 
        small part as 
        "system" and rest of enoumous part of the original microcanonical 
        system as "reservor" or "heat bath".
        By defining the system energy as $E_\mathrm{system}$ and reservor 
        energy as $E_\mathrm{reservor}$, we consider the energy exchange 
        between these two. Because total energy conserved, the sum of two 
        energy is constant value.
        $$ 
            E_\mathrm{system} + E_\mathrm{reservor} = E 
        $$

        The partition function is defined as the sum of all states
        $$
            Z = \sum_i e^{-\beta E}.
        $$
        Notice we also have the grand partition function which additionally
        condsiders chemical potentials.

        Using the partiton the functions, we are able obtain any operator. 
        A perticularly interesting value, we may obtain is the free energy,

        $$
            F = -\frac{1}{\beta}\log(Z).
        $$
        From here we may obtain the entropy, which is given by the 
        negative derivative of the free energy with respect to temperature
        $$
            S = -\frac{\partial F}{\partial T}.
        $$
        Another value of high importance, we may obtain from the partition
        function is the expectation value of the energy;
        $$
            \left<E\right> = -\frac{\partial\log(Z)}{\partial \beta}
        $$


        ## Ising Model (2d)

        spins $\pm 1$ on a lattice will self-organize to energertically favourable
        configurations. The energy of the system is given by

        $$
            H(\sigma) = -\sum_{\left<i j\right>}J_{ij}\sigma_i\sigma_j
        $$
        So it is energetically favourable to align with nearest neighbours. This happens over
        some timescale, so we iteratively pick a spin, assess the energy change, $\delta E$, though its flip
        and accept with a probability $p(\delta E)$.
        """)

    cols = st.columns(2)
    cols[0].markdown(r"""
        #### Metropolis algorithm

        The metropolis algorithm is a hard cutoff. So for values above a critical point
        are always accepted without considering probability. It makes computation easier.

        "Why?" you may ask, well; the large numbers arrising from the exponential
        are just that.
        """)
    
    cols[1].pyplot(metropolisVisualization())

    results, data = ising()
     
    st.markdown(r"""
        below are snapshots of the output of a simulation of the 2d Ising model
        using the metropolis algorithm.
        """)

    st.pyplot(plotSnapshots(nsnapshots = 4))

    cols = st.columns(2)
    cols[0].markdown(r"""If we track paramters through time,
        we may be able to spot a phase transition (they're a rare breed).
        On the right are plots of the energy and magnetization over time. Below
        is susceptibility as obtained the variance of the magnetization, 
        $\chi = \left< \left< M\right> - M\right>$ (:shrug)""")
    cols[1].pyplot(plotEnergy_magnetization())
    cols[0].pyplot(plotSusceptibility())
    
def run_phaseTransitions_CriticalPhenomena():
    st.markdown(r"""
    # Phase transitions & Critical phenomena

    ## mean-field approximation
    Assume a site is affected by the mean of the system as opposed
    to the actually interacting neighbours. This allows to solve this 
    type of problem analytically.

    This yields a mean-field partition functions
    $$
        Z_{MF} = \exp\left(-\beta\frac{NJzm^2}{2}
        \right)
        [2\cosh(Jzm\beta+h\beta)]^N
    $$
        """) 

    with st.sidebar:
        cols_sidebar =st.columns(2)
        size   = cols_sidebar[0].slider("size",0, 30, 10)
        J      = cols_sidebar[1].slider("J ",0.01, 2., 1.)
        nsteps = cols_sidebar[0].slider("Nsteps",0, 30, 10)
        beta   = cols_sidebar[1].slider("beta",0., 5., 1.) 
        cmap = st.select_slider('cmap', 
            ['inferno', 'gist_rainbow', 'RdBu', 'viridis',
            'inferno_r', 'magma'
            ])
        
    st.markdown(r'1D ising model')   
    
    
    chain = np.zeros(size) ; chain[chain<.5] = -1; chain[chain>=.5] = 1
    CHAINS = []
    for _ in range(nsteps):
        # pick random site
        i = np.random.randint(0,size-1)
        dE = (sum(chain[i-1:i+2])-chain[i])*chain[i]
        if np.random.rand()<np.exp(-beta*dE):
            chain[i] *= -1
        CHAINS.append(chain.copy())

    
    CHAINS = np.array(CHAINS)
    fig, ax = plt.subplots()
    ax.imshow(CHAINS, cmap=cmap, aspect = size/nsteps/3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_ylabel('Timestep', color='white')
    ax.set_xlabel('Site index', color='white')
    fig.patch.set_facecolor((.04,.065,.03))
    st.pyplot(fig)


    st.markdown(r"""## Transfer Matrix Method
    ...""")

def run_percolation_and_fractals():
    # Side bar
    with st.sidebar:
        st.markdown('### square grid percolation') 
        cols_sidebar = st.columns(2)
        size = cols_sidebar[0].slider('size', 10  , 100, 50)
        p = cols_sidebar[1].slider('p',       0.01, 1. , .1)
        marker_dict = {
            'point': '.',
            'square': 's',
            'pixel': ',',
            'circle': 'o',
        }
        marker_key = st.select_slider('marker', marker_dict.keys())
        marker = marker_dict[marker_key]
        seed = st.slider('seed',10,100)

        st.markdown('### Bethe lattice')

        st.markdown('### percolation on bethe lattice')

        st.markdown('### Fractals') 

    # Functions
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
                    visited.add((i,j))
        
        new_domains = {}
        index = 0
        for d in domains:
            if len(domains[d]) !=0:
                new_domains[index] = domains[d]
                index += 1
                
        return new_domains

    def percolation():
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
        fig.patch.set_facecolor('darkgrey')
        return fig

    def betheLattice():
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


    def run_fractals():
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
            fig.patch.set_facecolor('black')
            st.pyplot(fig)

        with st.sidebar:
            cols_sidebar = st.columns(2)
            logsize = cols_sidebar[0].slider(r'Resolution (log)',1.5,4., 3.)
            size = int(10**logsize)
            cols_sidebar[1].latex(r'10^{}\approx {}'.format("{"+str(logsize)+"}", size))
            cols_sidebar = st.columns(2)
            n = cols_sidebar[0].slider('n',1,50,27)
            a = cols_sidebar[1].slider('a',0.01,13.,2.3)

        res = stable(mandelbrot(makeGrid(size,  lims=[-1.85, 1.25, -1.25, 1.45]), a=a, n=n))
        plot_(res)

        cols = st.columns(2)
        cols[0].markdown(r"""
        The Mandelbrot set contains complex numbers remaining stable through
        
        $$z_{i+1} = z^a + c$$
        
        after successive iterations. We let $z_0$ be 0.
        """)
        cols[1].code(r"""
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
        return re+im*1j    """)

    # Render
    st.markdown(r"""# Percolation and Fractals""")

    st.markdown(r"""## Percolation""")
    st.pyplot(percolation())
    cols = st.columns(2)
    cols[0].markdown(r"""
    A matrix containing values between zero and one, with
    the value determining openness as a function of $p$.

    After generating a grid and a value for p, we look for 
    connected domains. 
    """)

    cols[1].code(r"""
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
                visited.add((i,j))""")

    st.markdown(r"""
    ## Bethe Lattice
    Bethe lattice (also called a regular tree)  is an infinite connected 
    cycle-free graph where all vertices have the same number of neighbors.  
    """)
    
    st.graphviz_chart(betheLattice())

    st.markdown(r"## Percolation on this lattice")


    run_fractals()

def run_random_walk():
    # Sidebar
    with st.sidebar:
        cols_sidebar = st.columns(2)
        nsteps = cols_sidebar[0].slider('nsteps',  4,   100, 14)
        seed   = cols_sidebar[1].slider('Seed',    0,   69 , 42)
        sigma2 = cols_sidebar[0].slider('Variance',0.2, 1. ,0.32)
        step_size = cols_sidebar[0].slider('Stepsize = random^x, x=', 0.,3.,0.)
        axisscale = cols_sidebar[1].radio('axis-scales', ['linear', 'symlog'])
        #yscale = cols_sidebar[1].radio('yscale', ['linear', 'symlog'])
    
    # Functions
    def accumulate(x):
        X=np.zeros(len(x)) ; X[0] = x[0]
        for i, _ in enumerate(x): X[i] = X[i-1]+x[i]
        return X

    def randomWalk(nsteps, sigma2=1, seed=42, axisscale='linear', step_size=0):
        (dx_f, dy_f) = (lambda theta, r=1: r*trig(theta) for trig in (np.cos, np.sin)) 
        dx_f = lambda theta, r = 1: r*np.cos(theta)
        dy_f = lambda theta, r = 1: r*np.sin(theta)

        np.random.seed(seed)
        thetas_uniform = np.random.uniform(0,2*np.pi,nsteps)
        thetas_randn = np.random.randn(nsteps)*sigma2
        thetas_bimodal = np.concatenate([ np.random.randn(nsteps//2) * sigma2-1, 
                                          np.random.randn(nsteps//2) * sigma2+1 ])

        thetas_uniform[0], thetas_randn[0], thetas_bimodal[0] = 0, 0, 0 

        rands = [thetas_uniform, thetas_randn, thetas_bimodal]
        rands_names = 'uniform, normal, bimodal'.split(', ')
        stepLengths = np.random.rand(nsteps).copy()**step_size

        def plot2():
            colors = 'r g y'.split()
            fig = plt.figure(figsize=(12,6))
            
            gs = GridSpec(3, 3, figure=fig)
            ax1 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
            ax2 = fig.add_subplot(gs[:, 1:])    
            
            lims = {'xl':0, 'xh':0, 'yl':0, 'yh':0}
            for i, (r, n, stepLength) in enumerate(zip(rands, rands_names, stepLengths)):
                dx, dy = dx_f(r, stepLength), dy_f(r, stepLength)
                dx = np.vstack([np.zeros(len(dx)), dx]).T.flatten()
                dy = np.vstack([np.zeros(len(dy)), dy]).T.flatten()
                
                ax1[i].plot(dx,dy, lw=1, c=colors[i])
                ax1[i].set(ylim=(-1,1), xlim=(-1,1), 
                            xticks=[], yticks=[],facecolor = "black",)
                
                x = accumulate(dx)
                y = accumulate(dy)
                ax2.plot(x,y, lw=2, label=n,c=colors[i])

            ax2.set(facecolor = "black",# xticks=[], yticks=[],
                    xticklabels=[],
                    xscale=axisscale,
                    yscale=axisscale)
            ax2.legend(fontsize=20)
                
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax1[0].set_title('Individual steps', fontsize=24)
            ax2.set_title('Cummulative path', fontsize=24)
            plt.tight_layout()
            fig.patch.set_facecolor('darkgrey')
            return fig
        return plot2()



    st.markdown(r"""# RandomWalk""")
    st.pyplot(randomWalk(nsteps,sigma2, seed, axisscale, step_size))

    cols = st.columns(2)
    cols[0].markdown(r"""
    Continous steps in a random direction illustrates the
    differences between diff. distributions.

    Red steps are generated by sampling theta from a uniform distribution.
    This tends to keep us close to the origin.

    Normal and bi-modal distributions are different in that the
    similarity of step direction causes great displacement.
    """)

    cols[1].code(r"""
def randomWalk(nsteps):
    for i in range(nsteps):
        theta = random()
        dx = np.cos(theta) ; x += dx
        dy = np.sin(theta) ; y += dy 
    """)

    st.markdown(r"""
    ## First return
    *Explore the time of first return in 1d, 2d and 3d*
        """)


def newNetwork():
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
    M = makeBetheLattice(34)
    domains = getDomains(M,0.6)
    open_arr = draw_from_matrix(M,domains)


def bereaucrats():
    st.markdown(r"# Beraucrats")


    def makeGrid(size):
        arr = np.zeros((size,size))
        return arr, size**2
    
    def fill(arr, N):
        rand_index = np.random.randint(0,N)
        who = (rand_index//arr.shape[1], rand_index%arr.shape[1])
        arr[who] += 1
        return arr


    def run():
        arr, N = makeGrid(size)
        results = {'mean':[], 'arr':[]}
        
        for step in range(nsteps):
            arr = fill(arr, N)  # bring in 1 task

            overfull_args = np.argwhere(arr>=4)  # if someone has 4 tasks redistribute to neighbours
            for ov in overfull_args:
                (i,j) = ov
                for pos in [(i+1, j), (i-1, j), (i,j+1), (i,j-1)]:
                    try: arr[pos] +=1
                    except: pass
                arr[i,j] -= 4
            results['mean'].append(np.mean(arr)) 
            results['arr'].append(arr.copy()) 

        return results

    with st.sidebar:
        cols_sidebar = st.columns(2)
        size = cols_sidebar[0].slider(r'size',5,40, 10)
        nsteps = cols_sidebar[1].slider('nsteps',1,5000,1000)
    
    results = run()

    # plot 1
    c = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart = st.line_chart()
    fig, ax = plt.subplots(1,5, figsize=(12,2.5))
    fig.patch.set_facecolor('black')
    a = [ax[idx].set(xticks=[], yticks=[], 
                facecolor='black') for idx in range(5)]
    
    steps = range(nsteps)[::nsteps//10]
    
    for val, (i, next) in enumerate(zip(steps, steps[1:])):
        
        progress_bar.progress(int((i + 1)/nsteps))  # Update progress bar.

        if val%2==0:  # plot imshow the grid
            idx = 0 if val==0 else idx+1 
            ax[idx].imshow(results['arr'][i], cmap="inferno")
            ax[idx].set(xticks=[], yticks=[])
            c.pyplot(fig)
            
        
        new_rows = results['mean'][i:next]

        # Append data to the chart.
        chart.add_rows(new_rows)

        # Pretend we're doing some computation that takes time.
        time.sleep(.1)

    status_text.text('Done!')

    st.markdown(r"""
    The problem with beraucrats, is that they dont finish tasks. When a task 
    lands on the desk of one, the set a side to start a pile. When that pile contains 
    4 tasks, its time to distribute them amongst the nieghbors. If a
    beraucrat neighbours an edge, they may dispose of the task headed in that direction. 
    """)


def bakSneppen():
    st.markdown(r"# Bak-Sneppen")
    def run(size, nsteps):
        chain = np.random.rand(size)

        X = np.empty((nsteps,size))
        L = np.zeros(nsteps)
        for i in range(nsteps):
            lowest = np.argmin(chain)  # determine lowest
            chain[(lowest-1+size)%size] = np.random.rand() # change left neighbour
            chain[lowest] = np.random.rand() # change ego
            chain[(lowest+1)%size] = np.random.rand() # change right neighbour
            X[i] = chain
            L[i] = np.mean(chain)

        fig, ax = plt.subplots()
        ax.imshow(X, aspect  = size/nsteps, vmin=0, vmax=1, cmap='gist_rainbow')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.patch.set_facecolor((.04,.065,.03))
        st.pyplot(fig)
        return L

    with st.sidebar:
        nsteps = st.slider('nsteps',1,30000,5000)
        size = st.slider('size',10,1000,300)

    L = run(size, nsteps)
    cols = st.columns(2)

    cols[0].markdown(r"""
    The Bak-Sneppen method starts with a random vector. At each
    timestep the smallest element and its two neighbors, are each 
    replaced with new random numbers.

    The figure on the right depicts the mean magnitude of elements in
    the vector.

    To build further on this, we should identify power laws along each dimension.
    """)

    fig, ax = plt.subplots()
    ax.plot(range(len(L)), L, c='purple')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(facecolor=(.04,.065,.03))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    cols[1].pyplot(fig)


def network():

    def makeBetheLattice(n_nodes = 10):
        M = np.zeros((n_nodes,n_nodes))

        idx = 1
        for i, _ in enumerate(M):
            if i ==0: n =3
            else: n =2
            M[i, idx:idx+n] = 1
            idx+=n

        return M+M.T

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
        color_map = ['r' if s==1 else 'white' for s in sick]
        
        pos = nx.nx_agraph.graphviz_layout(G) if len(pos)==0 else pos
        
        nx.draw_networkx(G,pos, node_color=color_map, edge_color='white')
        return pos


    with st.sidebar:
        network_type = st.selectbox('networt_type',['bethe', 'random'])
        N = st.slider('N',1,42,22)
        if network_type == 'random':
            a = st.slider('alpha', 0.,1.,0.97)
        
    fig, ax = plt.subplots()
    ax.set(facecolor=(.04,.065,.03))
    net = make_network(N,a) if network_type == 'random' else makeBetheLattice(N)
    draw_from_matrix(net)
    fig.patch.set_facecolor((.04,.065,.03))
    st.pyplot(fig)


def run_betHedging():

    st.markdown('# Bet-Hedghing')
    with st.sidebar:
        cols_sidebar = st.columns(2)
        nsteps = cols_sidebar[0].slider('nsteps',1,3000,500)
        starting_capital = cols_sidebar[1].slider('starting capital',1,1000,10)
        prob_loss = cols_sidebar[0].slider('loss probability', 0.,1.,.5) 
        invest_per_round = cols_sidebar[1].slider('invest per round', 0.,1.,.5) 

    capital = [starting_capital]
    for i in range(nsteps):
        if np.random.uniform()>prob_loss:
            capital.append(capital[i]*(1+invest_per_round))
        else:
            capital.append(capital[i]*(1-invest_per_round))

    fig, ax = plt.subplots()
    plt.plot(capital, c='purple')
    plt.xlabel('timestep', color='white')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(facecolor=(.04,.065,.03))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set(yscale='log')
    plt.ylabel('capital', color='white')
    st.pyplot(fig)


func_dict = {
    'Statistical Mechanics' : run_stat_mech,
    'Phase transitions & Critical phenomena' : run_phaseTransitions_CriticalPhenomena,
    'Percolation and Fractals'   : run_percolation_and_fractals,
	'RandomWalk'    : run_random_walk,
    'Bereaucrats'   : bereaucrats,
    'Bak-Sneppen'   : bakSneppen,
    #'new network'   : newNetwork,
    #'Networks'      : network,
    'Bet-Hedghing'  : run_betHedging,
    #'Bethe Lattice' : run_betheLattice
}

with st.sidebar:
	topic = st.selectbox("topic" , list(func_dict.keys()))

a = func_dict[topic] ; a()


#plt.style.available