import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
import numpy as np
import pandas as pd
from complex_utils import *
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

sys.setrecursionlimit(150000)

st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)

# setting matplotlib style:

mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['axes.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'
# mpl.rcParams['axes.grid'] = True  # should we?

textfile_path = 'assets/complex/text/'
def statisticalMechanics():
    
    # Sidebar
    with st.sidebar:
        size = st.slider('size',3,100,10)
        beta = st.slider('beta',0.01,5.,1.)
        nsteps = st.slider('nsteps',3,10000,100)
        nsnapshots = 9
    
    # Detailed Description
    ## some of these should perhaps be partially unpacked
    st.markdown(r"""# Statistical Mechanics""")
    text_dict = getText_prep(filename = textfile_path+'statisticalMechanics.md', 
                                split_level = 2)
    
    with st.expander("Microcanonical Ensemble", expanded=False):
        text_dict["Microcanonical Ensemble"]

    with st.expander("Ising Model ", expanded=False):
        text_dict["Ising Model "]
    
    with st.expander("Metropolis algorithm", expanded=False):
        text_dict["Metropolis algorithm"]
        cols = st.columns(2)
        cols[0].markdown(r"""
            As I mentioned earlier, first step of Metropolis algorithm is 
            trying random state transition.
            In second step of the algorithm, first caliculate energy difference.
            If energy difference is negative, accept the trial transition.
            If energy difference is positive, accept with weight 
            $\exp \left[-\beta \Delta E_{ij} \right]$. 
            """)
        
        cols[1].pyplot(metropolisVisualization(beta))

    with st.expander("Mean Field Solution to Ising Model", expanded=False):
        text_dict["Mean Field Solution to Ising Model"]
    
    # Simulation
    results, data = ising(size, nsteps, beta, nsnapshots)
     
    st.markdown(r"""
        ### Ising 2d Simulation  
        Snapshots of the output of a simulation of the 2d Ising modelusing the metropolis algorithm.
        """)
    st.pyplot(plotSnapshots(results, nsnapshots))

    st.markdown(r"""If we track paramters through time,
        we may be able to spot a phase transition (they're a rare breed).
        On the right are plots of the energy and magnetization over time. Below
        is susceptibility as obtained the variance of the magnetization, 
        $\chi = \left< \left< M\right> - M\right>$ (:shrug)""")
    st.pyplot(plotEnergy_magnetization(results))
    st.pyplot(plotSusceptibility(data))
    
def run_phaseTransitions_CriticalPhenomena():
    text_dict = getText_prep(filename = textfile_path+'phaseTransitions.md', 
                                split_level = 2)
    
    for key in text_dict:
        with st.expander(key, expanded=False):
            st.markdown(text_dict[key])
      
    with st.sidebar:
        size = st.slider('size',3,100,10)
        beta = st.slider('beta',0.01,5.,1.)
        nsteps = st.slider('nsteps',3,10000,100)
        nsnapshots = 4
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
    ax.imshow(CHAINS, #cmap=cmap, 
    aspect = size/nsteps/3)
    ax.set_ylabel('Timestep', color='white')
    ax.set_xlabel('Site index', color='white')
    st.pyplot(fig)


    st.markdown(r"""
    ## Transfer Matrix Method
    ...
    """)

def percolation_and_fractals():
    # Side bar
    with st.sidebar:
        st.markdown('## Paramteres') 
        with st.expander('square grid percolation'):

            cols_sidebar = st.columns(2)
            size = cols_sidebar[0].slider('size', 4  , 64, 42)

            marker_dict = {'point': '.','pixel': ',',}
            marker = marker_dict[st.radio('marker', marker_dict.keys())]
            seed = cols_sidebar[1].slider('seed',10,100)

        with st.expander('Mandelbrot'):
            cols_sidebar = st.columns(2)
            logsize = cols_sidebar[0].slider(r'Resolution (log)',1.5,4., 3.)
            size_fractal = int(10**logsize)
            cols_sidebar[1].latex(r'10^{}\approx {}'.format("{"+str(logsize)+"}", size_fractal))
            cols_sidebar = st.columns(2)
            n = cols_sidebar[0].slider('n',1,50,27)
            a = cols_sidebar[1].slider('a',0.01,13.,2.3)


    # Render
    st.markdown(r"""# Percolation and Fractals""")
    
    st.markdown(r"""## Percolation""")
    cols = st.columns(2)

    cols[0].markdown(r"""
    A matrix containing values between zero and one is generated. Values greater than $p$ are *open*. On the right, is a randomly generated grid. Vary $p$ to alter openness to affect the number of domains, $N(p)$.
    """)
    
    p_percolation = cols[0].slider("""p =""",       0.01, 1. , .1)
    fig_percolation, domains = percolation(size, seed, p_percolation,marker)
    cols[1].pyplot(fig_percolation)
    st.latex(r"""N({}) = {}""".format(p_percolation, len(domains)))
    
    st.markdown(r"""We may visualize this relation:""")
    def percolation_many_ps(n_ps=10):
        Ns = {}
        for p_ in np.linspace(0.01,.9,n_ps):
            _, domains = percolation(size, seed, p_,marker)
            Ns[p_] = {'number of domains':len(domains),
                        'domain sizes' : [len(domains[i]) for i in domains]
                    }
        
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(Ns.keys(),[Ns[i]['number of domains'] for i in Ns] , c='white')
        ax.set_xlabel(r'$p$', color='white')
        ax.set_ylabel(r'Number of domains, $N$', color='white')
        st.pyplot(fig)

        #max_domain_size = np.max([np.max(Ns[i]['domain sizes']) for i in Ns])
        
        #bins =np.logspace(0,np.log10(max_domain_size),10)
        #hists = np.array([np.histogram(Ns[i]['domain sizes'], bins=bins)[0] for i in Ns])
        #fig, ax = plt.subplots(figsize=(8,3))

        
        #hists_normalized = hists/np.max(hists, axis=0)
        #for h in hists:
        #    ax.plot(bins[:-1], h)
        #ax.set(xscale='log')
        #ax.set_xlabel(r'$p$', color='white')
        #ax.set_ylabel(r'Number of domains, $N$', color='white')
        #st.pyplot(fig)

    percolation_many_ps(10)


    # Bethe lattice
    st.markdown(r"""
    ## Bethe Lattice""")
    cols = st.columns(2)
    cols[0].markdown(r"""
    Bethe lattice (also called a regular tree)  is an infinite connected 
    cycle-free graph where all vertices have the same number of neighbors.  
    
    
    We may perform percolation on this lattice. To do this, we fill the adjencancy matrix, not with boolean value, but instead with random samples drawn from a uniform distribution.
    """)
    cols[1].pyplot(betheLattice(0, size=10))

    #st.graphviz_chart(betheLattice_old())
    size_beth = 62
    p_beth = st.slider("""p = """,       0.01, 1. , .5)
    st.pyplot( betheLattice(p_beth, size=size_beth))
    
    st.markdown(r'''Again we may take a look at the number of domains as a function of $p$.''')

    Ns = betheLattice(size=32, get_many=True, 
                    ps=np.linspace(.1,.9,10))
        
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(Ns.keys(),Ns.values() , c='white')
    ax.set_xlabel(r'$p$', color='white')
    ax.set_ylabel(r'Number of domains, $N$', color='white')
    st.pyplot(fig)


    st.markdown(r"""## Mandelbrot""")
    st.markdown(r"""
    The Mandelbrot set contains complex numbers remaining stable through the mandelbrot function after successive iterations. Note; we let $z_0$ be 0. The two main essential pieces of code are displayed below the plot.
    """)
    st.pyplot(run_fractals(size_fractal, a, n))
    st.markdown(r"""
    To optimize run-time, we have used that the output is symmetric across the real axis. We only calculate one side, i.e., 
    $$
        \text{stable} \left(\text{mandelbrot}(a+ib)\right)
        =
        \text{stable} \left(\text{mandelbrot}(a-ib)\right)
    $$
    """)
    
    cols = st.columns(2)
    cols[0].code(r"""
def stable(z):
    try:
        return False if abs(z) > 2 else True
    except OverflowError:
        return False
stable = np.vectorize(stable)""")
    
    cols[1].code(r"""
def mandelbrot(c, a=2, n=50):
    z = 0
    for i in range(n):
        z = z**a + c
    return z
""")
    st.markdown(r"""## Fractal Dimension""")

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
            ax2.set_title('Cumulative path', fontsize=24)
            plt.tight_layout()
            fig.patch.set_facecolor('darkgrey')  # do we want black or darkgrey??
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

    # 1d
    def run_firstReturn1D():
        lengths = []
        lines = {}
        c=st.empty()
        for idx in range(100):
            

            x = [0] 
            for i in range(100):
                change = -1 if np.random.rand()< 0.5 else 1
                x.append(x[i]+change)
                if x[i+1] == 0: break
            lines[idx] = x

            fig, ax = plt.subplots(1,2)
            for idx in lines.keys():
                x = lines[idx]
            
                ax[0].plot(x, range(len(x)))#, c='orange')
            ax[0].set_xlabel('x position', color='white')
            ax[0].set_ylabel('time', color='white')
            ax[0].set(xticks=[0], yticks=[])
            ax[0].grid()

            lengths.append(len(x))

            ax[1].hist(lengths)
            ax[1].set_xlabel('First return time', color='white')
            ax[1].set_ylabel('occurance frequency', color='white')
            #ax[1].set(xticks=[0], yticks=[])
            ax[1].grid()
            c.pyplot(fig)

    a = st.button('run_firstReturn1D')
    if a: run_firstReturn1D()

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
    net = make_network(N,a) if network_type == 'random' else makeBetheLattice(N)
    draw_from_matrix(net)
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
    ax.set(yscale='log')
    plt.ylabel('capital', color='white')
    st.pyplot(fig)


func_dict = {
    'Statistical Mechanics' : statisticalMechanics,
    'Phase transitions & Critical phenomena' : run_phaseTransitions_CriticalPhenomena,
    'Percolation and Fractals'   : percolation_and_fractals,
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
