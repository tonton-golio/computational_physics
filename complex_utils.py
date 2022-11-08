import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
import time
import pandas as pd
try: import networkx as nx # having trouble with this when hosted
except: pass
import sys; sys.setrecursionlimit(150000)

# Stremlit layout
st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
	layout="centered", 
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

# text parsing
textfile_path = 'assets/complex/text/'
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

def text_expander(key, text_dict, expanded=False):
    with st.expander(key, expanded=expanded):
        st.markdown(text_dict[key])

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
    
    fig, ax = plt.subplots(nsnapshots//4,4, figsize=(9,9))
    if nsnapshots == 4: ax = np.vstack([ax,ax])
    for idx, key in enumerate(results['snapshots'].keys()):
        ax[idx//4, idx%4].imshow(results['snapshots'][key], cmap='BuGn')
        ax[idx//4, idx%4].set(xticks=[], yticks=[]) 
        ax[idx//4, idx%4].set_title(key, color="white")
        plt.tight_layout()
    plt.close()
    return fig

def plotEnergy_magnetization(results):
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    ax2 = ax.twinx()
    ax.plot(results['Energy'],c='red', lw=2)

    ax2.plot(np.abs(results['Magnetization']), color='orange')
    
    ax.set_xlabel('Timestep', color='white')
    ax.set_ylabel('Energy', color='red')
    ax2.set_ylabel('|Magnetization|', color='orange')
    plt.close()
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
    plt.close()
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
    plt.close()
    return fig


# Percolation and Fractals
def percolation(size=10, seed=69, p=0.4,marker='.', devmod=False):
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
                    domain.add(n) ; visited.add(n)
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
                    
                    visited = visited.union(visited_)
                    if len(domain) > 0:
                        domains[index] = domain
                        index+=1
                else: visited.add((i,j))
        
        new_domains = {}
        index = 0
        for d in domains:
            if len(domains[d]) !=0:
                new_domains[index] = domains[d]
                index += 1
                
        return new_domains

    grid = makeGrid(size,seed)
    domains = getDomains(grid, p)
    #if devmod: return domains
    x = np.arange(size+2)
    X,Y = np.meshgrid(x,x)
    xx = np.concatenate([list(domains[i]) for i in domains])
    
    if devmod:
       return domains, xx
    else:
        fig, ax = plt.subplots()
        # background
        ax.scatter(X,Y, c='black')

        # colors
        colors = sns.color_palette("hls", len(domains))
        np.random.shuffle(colors)
        colors = np.concatenate([[colors[i]]*len(domains[i]) for i in domains])

        # plot
        ax.scatter(xx[:,0], xx[:,1], c=colors, marker=marker)
        ax.set(xticks = [], yticks = [], facecolor='black')
        plt.close()
        
        return fig, domains

def percolation_many_ps(n_ps, size, seed):
        Ns = {}
        for p_ in np.linspace(0.01,.9,n_ps):
            domains, __ = percolation(size, seed, p_, devmod=True)
            Ns[p_] = {'number of domains':len(domains),
                        'domain sizes' : [len(domains[i]) for i in domains]
                    }
        
        fig, ax = plt.subplots(figsize=(5,2))
        ax.plot(Ns.keys(),[Ns[i]['number of domains'] for i in Ns] , c='white')
        ax.set_xlabel(r'$p$', color='white')
        ax.set_ylabel(r'Number of domains, $N$', color='white')
        plt.close()
        return fig

def animate_many_percolations(size=30 , steps = 10, filename='animation.gif', fps=5):
    def many_perc(size = 30, low=0.01, high=0.9, steps=10, seed=42, marker='.'):
        out = {}
        for p in np.linspace(low, high, steps):
            domains, xx = percolation(size, seed, p,marker, devmod=True)#[0]
            N = len(domains)
            sizes = np.array([len(domains[i]) for i in domains])#.reshape(-1,1)
            
            idx = 1 if p> 0.3 else 0
            idx = 2 if p> 0.7 else idx 
            
            out[p] = {'domains' : domains, 
                        'sizes' : sizes, 
                        'xx' : xx}
        return out

    def animate(out, filename='animation.gif', fps=5):
        fig = plt.figure(constrained_layout=True)

        gs = GridSpec(2, 3, figure=fig)
        ax = [fig.add_subplot(gs[0, 0]), 
            fig.add_subplot(gs[1, 0])]
        ax_im = fig.add_subplot(gs[:, 1:])

        camera = Camera(fig)

        for i, p in enumerate(out.keys()):
            
            # histogram
            counts, bins = np.histogram(out[p]['sizes'])
            colors = sns.color_palette("gray", len(out.keys()) )
            bin_mids = (bins[:-1] + bins[1:])/2
            
            ax[0].stairs(counts*1, bins, color=colors[i], label=p, lw=3)
            ax[1].stairs(counts*bin_mids, bins, color=colors[i], label=p, lw=3)
            ax[1].set_xlabel(r'Domain size, $s$', color="white")
            ax[0].set_ylabel(r'Ocurrance freq., $f$', color="white")
            ax[1].set_ylabel(r'Weight, $w=f\times s$', color="white")
            ax[0].legend([f'p={round(p, 2)}'], facecolor='beige')
            ax[0].set(yscale='log')
            ax[1].set(yscale='log')
            
            # imshow
            domains = out[p]['domains']
            colors_im = sns.color_palette("hls", len(domains))
            if len(domains) > 20: np.random.shuffle(colors_im)
            colors_im = np.concatenate([[colors_im[i]]*len(domains[i]) for i in domains])
            
            xx = out[p]['xx']
            ax_im.scatter(xx[:,0], xx[:,1], c=colors_im, marker='.')
            ax_im.set(xticks = [], yticks = [], facecolor='black')
            
            plt.tight_layout()
            camera.snap()

        #plt.xscale('log')
        animation = camera.animate()
        animation.save(filename, fps=fps)


    out = many_perc(size = size, steps=steps)
    animate(out, filename=filename, fps=fps)

def betheLattice(p=0.1, size=62, get_many=False, ps=[.5], degree=3):
    def makeBetheLattice(n_nodes = 10, degree=3):
        M = np.zeros((n_nodes,n_nodes))

        idx = 1
        for i, _ in enumerate(M):
            if i ==0: n =degree
            else: n = degree-1
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
        color_map = ['black' if i not in inDomain.keys() else palette[inDomain[i]] for i in range(len(M))]

        if len(palette) == 0: color_map = ['orange'] * len(M)

        fig = plt.figure()
        nx.draw_networkx(G, node_color=color_map, pos=nx.kamada_kawai_layout(G))
        plt.close()
        return fig
        
    if get_many == False:
        M = makeBetheLattice(size, degree=degree)
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
        try: return False if abs(z) > 2 else True
        except OverflowError: return False
    stable = np.vectorize(stable)

    def mandelbrot(c, a, n=50):
        z = 0
        for i in range(n): z = z**a + c
        return z

    def makeGrid(resolution, lims=[-1.85, 1.25, -1.45, 1.45]):
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
        plt.close()
        return fig

    res = stable(mandelbrot(makeGrid(size_fractal,  lims=[-1.85, 1.25, -1.3, 1.3]), a=a, n=n))
    return plot_(res), res

def fractal_dimension(res):
    box_sizes_plot = [8,4,2]
    fig, ax = plt.subplots(1,len(box_sizes_plot))

    SQUARES = {}
    box_sizes = np.arange(2,32)
    for box_size in box_sizes:
    
        squares = np.zeros(np.array(res.shape)//box_size)
        
        for i, _ in enumerate(squares):
            for j, _ in enumerate(squares[i]):
                select = res[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                if (0 in select) and (1 in select):
                    squares[i,j] = 1

        SQUARES[box_size] = squares
    
    for i, box_size in enumerate(box_sizes_plot):
        ax[i].imshow(SQUARES[box_size].T)
    st.pyplot(fig)
    plt.close()


    fig2 = plt.figure()
    plt.plot(box_sizes, [1*np.sum(SQUARES[box_size]) for box_size in box_sizes])
    st.pyplot(fig2)


# Phase Transitions and Critical Phenomena
def ising_1d(size, beta, nsteps):
    chain = np.zeros(size) ; chain[chain<.5] = -1; chain[chain>=.5] = 1
    chain_MF = chain.copy()

    # normal approach
    CHAINS, dEs = [], []
    for _ in range(nsteps):
        # pick random site
        i = np.random.randint(0,size-1)
        dE = (sum(chain[i-1:i+2])-chain[i])*chain[i]
        if np.random.rand()<np.exp(-beta*dE):
            chain[i] *= -1
        CHAINS.append(chain.copy())
        dEs.append(dE)
    CHAINS = np.array(CHAINS)

    # MF
    CHAINS_MF, dEs_MF = [], []
    J, z, N = 1,2, len(chain_MF)
    for _ in range(nsteps):
        # pick random site
        i = np.random.randint(0,size-1)
        si = chain_MF[i]
        m = np.mean(chain_MF)
        dE = m * si
        if np.random.rand()<np.exp(-beta*dE):
            chain_MF[i] *= -1
        CHAINS_MF.append(chain_MF.copy())
        dEs_MF.append(dE)
    CHAINS_MF = np.array(CHAINS_MF)
    
    fig, ax = plt.subplots(2,2, sharex=True)
    ax[0,0].imshow(CHAINS.T, cmap='copper', aspect = nsteps/size/3)
    ax[1,0].set_xlabel('Timestep', color='white')
    ax[0,0].set_ylabel('Site index', color='white')
    
    ax[0,1].imshow(CHAINS_MF.T, cmap='copper', aspect = nsteps/size/3)
    ax[1,1].set_xlabel('Timestep', color='white')
    ax[1,0].set_ylabel(r'$\Delta E$', color='white')

    ax[1,0].plot(dEs)
    ax[1,1].plot(dEs_MF)
    
    plt.close()
    return fig, CHAINS


# SOC
def bakSneppen(size, nsteps):
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
    plt.close()
    return fig, L

def bakSneppen_plot_fill(L):
    fig, ax = plt.subplots()
    ax.plot(range(len(L)), L, c='purple')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(facecolor=(.04,.065,.03))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.close()
    return fig

def accumulate(x):
    X=np.zeros(len(x)) ; X[0] = x[0]
    for i, _ in enumerate(x): X[i] = X[i-1]+x[i]
    return X

def randomWalk_2d(nsteps, sigma2=1, seed=42, axisscale='linear', step_size=0):
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
    st.markdown(r"""
    Continous steps in a random direction illustrates the
    differences between diff. distributions.

    Red steps are generated by sampling theta from a uniform distribution.
    This tends to keep us close to the origin.

    Normal and bi-modal distributions are different in that the
    similarity of step direction causes great displacement.
    """)
    return plot2()

def firstReturn1D(nsteps=1000, nwalks=50):
    lengths = []
    lines = {}
    c=st.empty()
    for idx in range(nwalks):
        
        x = [0] 
        for i in range(nsteps):
            change = -1 if np.random.rand()< 0.5 else 1
            x.append(x[i]+change)
            if x[i+1] == 0: break
        lines[idx] = x

        fig, ax = plt.subplots(1,2, figsize=(8,3))
        for idx in lines.keys():
            x = lines[idx]
        
            ax[0].plot(x, range(len(x)))#, c='orange')
        ax[0].set_xlabel('x position', color='white')
        ax[0].set_ylabel('time', color='white')
        ax[0].set(xticks=[0], yticks=[])
        ax[1].set(xticks=[0, nsteps//4, nsteps//2, nsteps//4*3, nsteps],
                xticklabels=[0, nsteps//4, nsteps//2, nsteps//4*3, f'>{nsteps}'])
        ax[0].grid()

        lengths.append(len(x))

        ax[1].hist(lengths)
        ax[1].set_xlabel('First return time', color='white')
        ax[1].set_ylabel('occurance frequency', color='white')
        #ax[1].set(xticks=[0], yticks=[])
        ax[1].grid()
        c.pyplot(fig)
    plt.close()

def firstReturn2D(nsteps=4000, nwalks=20):
    lengths = []
    lines = {}
    c=st.empty()  # empty for plots
    for idx in range(nwalks):  # walks
        x, y = np.zeros(nsteps+1), np.zeros(nsteps+1) 

        dx_func = lambda theta, r = 1: r*np.cos(theta)
        dy_func = lambda theta, r = 1: r*np.sin(theta)

        thetas = np.random.uniform(0,2*np.pi,nsteps)
        
        #stepLengths = np.random.rand(nsteps).copy()**step_size
        has_left_unit_circle = False
        for i, theta in enumerate(thetas):
            dx = dx_func(theta)
            dy = dy_func(theta)
            x[i+1] = x[i]+dx
            y[i+1] = y[i]+dy
            if has_left_unit_circle == False:
                if (x[i+1]**2 + y[i+1]**2)**.5 > 3/2:
                    has_left_unit_circle = True
            else:
                if (x[i+1]**2 + y[i+1]**2)**.5 < 1:
                    x = x[:i+2]
                    y = y[:i+2]
                    break
        lines[idx] = {'x':x, 'y':y}
        
        fig, ax = plt.subplots(1,2, figsize=(8,3))
        for idx in lines.keys():
            ax[0].plot(lines[idx]['x'], lines[idx]['y'])

        ax[0].set_xlabel('x position', color='white')
        ax[0].set_ylabel('y position', color='white')
        ax[0].set(xticks=[0], yticks=[0])
        ax[1].set(xticks=[0, nsteps//4, nsteps//2, nsteps//4*3, nsteps],
                xticklabels=[0, nsteps//4, nsteps//2, nsteps//4*3, f'>{nsteps}'])
        ax[0].grid()

        lengths.append(len(lines[idx]['x']))

        ax[1].hist(lengths)
        ax[1].set_xlabel('First return time', color='white')
        ax[1].set_ylabel('occurance frequency', color='white')
        #ax[1].set(xticks=[0], yticks=[])
        ax[1].grid()
        c.pyplot(fig)
    plt.close()

def bereaucrats(nsteps, size=20):

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


# Networks
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
    
    #pos = nx.nx_agraph.graphviz_layout(G) if len(pos)==0 else pos
    
    fig, ax = plt.subplots()
    nx.draw_networkx(G, node_color=color_map, edge_color='white')
    return fig


# Agents
def game_of_life(size=6, nsteps=4, initial_config = 'boat'):
    """
    If the cell is alive, then it stays alive if it has either 2 or 3 live neighbors
    If the cell is dead, then it springs to life only in the case that it has 3 live neighbors
    """
    initially_filled = {
                        'boat' : [(2,2), (3,2), (2,3), (4,3), (3,4)],
                        'ship' : [(2,2), (3,2), (2,3), (4,3), (4,4), (3,4)],
                        'square' : [(2,2), (3,2), (2,3), (3,3)],
                        'loaf' : [(2,1), (3,1), (1,2), (4,2), (2,4), (1,3), (3,3)],
                        'glider' : [(2,2), (3,2), (2,3), (4,3), (2,4), ],
        }[initial_config]

    def initGrid(size, initially_filled):
        grid = np.zeros((size,size), dtype=bool)
        for i in initially_filled: grid[i] = True
        return grid

    def sumNeighbors(grid, i,j):
        S = 0
        for pos in [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                    (i-1, j-1), (i+1, j+1), (i+1, j-1), (i-1, j+1)]:
            if (pos[0] >= 0) and (pos[1] >= 0) and (pos[0] < len(grid)) and (pos[1] < len(grid)):
                    S += grid[pos]
        return S

    def update(grid):
        grid_new = grid.copy()
        for i in range(size):
            for j in range(size):
                if grid[i,j]: # if ego is alive
                    if sumNeighbors(grid, i, j) not in [2,3]:
                        grid_new[i,j] = False
                else:
                    if sumNeighbors(grid, i, j) ==3:
                        grid_new[i,j] = True

        return grid_new


    grid = initGrid(size, initially_filled)

    fig, ax = plt.subplots(1,5)
    ax[0].imshow(grid)
    ax[0].set_title('initial', color='white')

    grid = update(grid)
    ax[1].imshow(grid)
    ax[1].set_title('after 1 itr', color='white')

    grid = update(grid)
    ax[2].imshow(grid)
    ax[2].set_title('after 2 itr', color='white')

    grid = update(grid)
    ax[3].imshow(grid)
    ax[3].set_title('after 3 itr', color='white')

    grid = update(grid)
    ax[4].imshow(grid)
    ax[4].set_title('after 4 itr', color='white')

    _ = [ax[i].set(xticks=[], yticks=[]) for i in range(len(ax))]
    plt.tight_layout()
    
    plt.close()
    return fig


# Econophysics
def betHedging(p, noise, invest_per_round, nsteps, win_multiplier=2, loss_multiplier=.5):
    capital = [1]
    
    for i in range(nsteps):
        prob_loss = 1/(2*(1-p))+np.random.uniform(-1,1,None)*noise
        if np.random.uniform()>prob_loss:
            capital.append(capital[i]*win_multiplier)
        else:
            capital.append(capital[i]*loss_multiplier)

    fig, ax = plt.subplots()
    plt.plot(capital, c='purple')
    plt.xlabel('timestep', color='white')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(yscale='log')
    plt.ylabel('capital', color='white')
    plt.close()
    return fig
