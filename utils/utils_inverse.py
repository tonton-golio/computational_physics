import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
from time import sleep
import matplotlib as mpl
from scipy.constants import gravitational_constant


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import streamlit as st
from numba import prange, jit
import pandas as pd
from time import time


def set_rcParams():
	"""
		setting matplotlib style
	"""
	# extra function
	mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['axes.facecolor'] = 'grey'
	mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['xtick.color'] = 'white'
	mpl.rcParams['ytick.color'] = 'white'
	# mpl.rcParams['axes.grid'] = True  # should we?
	mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'

set_rcParams()

# General

text_path = 'assets/inverse_problems/text/'

st.set_page_config(page_title="Inverse Problems", 
    page_icon="🧊", 
    #layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)

def getText_prep(filename = text_path+'week1.md', split_level = 2):
    with open(filename,'r', encoding='utf8') as f:
        file = f.read()
    level_topics = file.split('\n'+"#"*split_level+' ')
    text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
    return text_dict  

def template():
    st.title('')

    # Main text
    text_dict = getText_prep(filename = text_path+'bounding_errors.md', split_level = 2)

    st.markdown(text_dict["Header 1"])
    st.markdown(text_dict["Header 2"])
    with st.expander('Go deeper', expanded=False):
        st.markdown(text_dict["Example"])


# Week 1

def G_ij(zi, xj): 
    return gravitational_constant * np.log( ((zi+1)**2 + xj**2) / ( zi**2 + xj**2 ) )

def G_matrix(xs, zs):
    G = np.array([[G_ij(z, x) for z in zs] for x in xs])
    return G


def contour_of_G(G, xlabel='$x$-position', ylabel='depth'):
    fig, ax = plt.subplots( figsize=(5,3))
    CS = ax.contourf(G, 10, cmap=plt.cm.bone)

    CS2 = ax.contour(CS, levels=CS.levels[::2], colors='r')

    ax.set_title(r'$G$')
    ax.set_ylabel(ylabel, color='black')
    ax.set_xlabel(xlabel, color='black')

    cbar = fig.colorbar(CS)
    cbar.add_lines(CS2)
    plt.gca().invert_yaxis()

    fig.set_facecolor('lightgray')
    plt.tight_layout()
    plt.close()
    return fig 

def getParams(G, d_obs, eps_space = np.logspace(-12, -10, 200)):
    ms = []
    
    for epsilon in eps_space:
        m_e = np.linalg.inv(G.T@G + epsilon**2 * np.eye(100) ) @  (G.T @d_obs)

        ms.append(m_e)

    return np.array(ms)


def find_minimum(G, ms, d_obs, 

    eps_space,
    data_error = [10**(-9)] * 18,
    data_error1 = [10**(-8)] * 18):

    result = [ abs( np.linalg.norm(G @ m - d_obs) -
                 np.linalg.norm(data_error) 
               ) for m in ms ]

    result1 = [ abs( np.linalg.norm(G @ m - d_obs) -
                 np.linalg.norm(data_error1) 
               ) for m in ms ]

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(eps_space, result, label='data error = 10e-9')
    ax.plot(eps_space, result1, label='data error = 10e-8')

    min_eps = eps_space[np.argmin(result)]
    min_eps1 = eps_space[np.argmin(result1)]
    ax.axvline(min_eps,ls='--',label=f'minimum = {round(min_eps,14)}')
    ax.axvline(min_eps1,ls='--',c='orange', label=f'minimum = {round(min_eps1,14)}')

    ax.set_xlabel(r'$\epsilon$', color='black')
    ax.set_ylabel(r'Deviation', color='black')

    ax.set(
            xscale='log', yscale='log', 
            #xlim=(1e-15, 1e-11)
        )

    ax.legend(facecolor='beige')
    ax.set_facecolor('lightgray')
    fig.set_facecolor('lightgray')

    plt.close()
    return fig


def ass2():
    
    plt.style.use('dark_background')

    def make_m0(n_slabs = 5):

        heights = np.random.uniform(2000,10000,n_slabs)

        zs = np.array([sum(heights[:i+1]) for i in range(len(heights))])
        rhos = np.random.uniform(-2000,2000,n_slabs)
        
        m = np.vstack([zs, rhos])
        return m

    def check_m(m):
        # check if m is valid
        'aka: g(m)'
        zs, rhos = m[0], m[1]
        heights = np.concatenate([zs[:1], zs[1:]-zs[:-1]])
        #print(zs,'\n', heights)
        assert ( heights<=10000).all()
        assert ( heights>=2000).all()

        assert ( rhos<=2000).all()
        assert ( rhos>=-2000).all()

    def alter_m(m, sig_h, sig_rho):
        # draw new m
        new_good, itr = False, 0
        while not new_good:
            try:
                m_new = np.vstack([np.random.normal(m[0], sig_h), 
                                    np.random.normal(m[1], sig_rho)])
                check_m(m)
                new_good=True
                break
            except: pass
            if itr > 100: break
            else: itr += 1
        return m_new


    def getEstimate(m, x):
        'aka: g(m)'
        zs, rho = m[0], m[1]
        #print(zs, rhos)

        zs_with0 = np.hstack([[0], zs])
        d_est = np.array([sum(r*np.log( (b**2 + xj**2)/(t**2+xj**2) ) for r, b,t in zip(rho, zs, zs_with0)) for xj in x])
        d_est *= gravitational_constant
        return d_est

    def loss(d_obs, d_est, K1=1, s=10e-9):  # this mitigates overflow
        return  -1*np.log(K1) + (np.linalg.norm(d_obs-d_est)**2) / (2*s**2) 

    def MCMC(x, d_obs, n_init=10, n_steps=100, beta = 20, sig_h=30, sig_rho=10, verbose=True):
        res = {}
        if verbose:
            progress_cols=st.columns(2)
            my_bar = progress_cols[0].progress(0.0)
            start = time()
            progress_info = progress_cols[1].empty()
            progress_info.write(f"Elapsed time = {0}, itr = {0} / {n_init}, ETA = {'?'}")

        for run_ in prange(n_init):
            if verbose: 
                my_bar.progress(run_/n_init)
                run_time = time()-start
                time_per_run = 1 if run_ ==0 else run_time/run_
                progress_info.write(f"Elapsed time = {round(run_time,1)}, itr = {run_} / {n_init}, ETA = {round(time_per_run*(n_init-run_),2)}")
            ms = []
            losses = []
            acceptance = []

            m =  make_m0()
            d_est = getEstimate(m , x)
            loss_ = loss(d_obs, d_est)
            
            for i in range(n_steps):
                
                m_new = alter_m(m, sig_h*loss_**.5+2, sig_rho*(loss_+1)**.5+2)
                
                d_est_new = getEstimate(m_new , x)
                loss_new = loss(d_obs, d_est_new)

                if loss_new < loss_:                     # accecpt change
                    m, d_est, loss_ = m_new.copy(), d_est_new.copy(), loss_new.copy()
                    acceptance.append(1)
                else:
                    acc_p = np.exp(-loss_new-loss_*beta)
                    if np.random.uniform(0,1) < acc_p:   # accept
                        m, d_est, loss_ = m_new.copy(), d_est_new.copy(), loss_new.copy()
                        acceptance.append(1)
                    else:
                        acceptance.append(0)
                
                losses.append(loss_)
                ms.append(m)
            res[run_] = {
                    'losses' : np.array(losses),
                    'ms' : np.array(ms),
                    'acc' : np.array(acceptance)
                    }
        if verbose:
            my_bar.progress(1.0)
            run_time = time()-start
            progress_info.write(f"Elapsed time = {round(run_time,1)}, itr = {n_init} / {n_init}, ETA = {0}")
        #np.savez(f'assets/ass2_results/{n_init}_{n_steps}_{beta}_{sig_h}_{sig_rho}', res)
        return res

    def load_data(plot=True, st=st):
        data = np.genfromtxt('assets/inverse_problems/data/gravdata_ass2.txt')
        x, d_obs = data[:,0], data[:,1]

        if plot:
            fig = plt.figure(figsize=(5,3))
            plt.scatter(x, d_obs, marker='x', label='observations')
            plt.close()
            st.pyplot(fig)
                
        return x, d_obs

    def show():
        pass

    def settings_menu():
        pass
        # here we should be able to vary all the parameters of MCMC

    st.markdown('# Height and density of slabs')
    cols = st.columns(2)
    img = plt.imread('assets/inverse_problems/images/diagram_fault.png')

    cols[1].image(img)

    x, d_obs = load_data(plot=True, st = cols[1])

    cols[0].markdown(r"""
    We have made 18 measurements of the horizontal gravity gradient above and to the right of a vertical fault (fracture). We want to predict the height of slaps in the earth below, as well their relative densities.

    The horizontal gravity gradient at one x-location is given by;
    $$
    d_j = \frac{\partial g}{\partial x} (x_j) = 
    \int_0^\infty \frac{2Gz}{x_j^2+z^2} \Delta\rho(z)dz.
    $$
    Which, for computation, we need to convert into a specific sum;
    $$
    d_j = G\sum_i^n \Delta\rho_i \log\left(
    \frac{(z_\text{base}^i)^2 + x_j^2}{(z_\text{top}^i)^2 + x_j^2}
    \right).
    $$
    """)


    st.markdown('---')

    st.markdown("""
    Since we don't have any idea how thick or how dense these layers may be, we initialize many MCMC walkers distributed uniformly in the permissable range.

    Additionally, we set the learning-rate to be proportional to the root of the loss, thus allowing us to quickly exit high-loss regions.
    """)

    mode = 'run'#st.sidebar.select_slider('mode', ['run', 'load'][::-1])
    if mode == 'run':
        cols = st.columns(6)
        n_init = cols[0].select_slider('n_init', np.logspace(0,2,10,dtype=int))
        n_steps= cols[1].select_slider('n_steps', np.logspace(0,4,20,dtype=int)) 
        beta =   cols[2].select_slider('beta',     np.round(np.logspace(-3,1.5,10), 3)) 
        sig_h=   cols[3].select_slider('sig_h',     np.round(np.logspace(-3,1.5,10), 3)) 
        sig_rho= cols[4].select_slider('sig_rho', np.round(np.logspace(-3,1.5,10), 3)) 
        burn = cols[5].slider('burn', 10,10000,400)


        data = MCMC(x, d_obs, n_init, n_steps, beta, sig_h, sig_rho)
    else:
        file = np.load('assets/ass2_results/10000_1000_20_30_10.npz', allow_pickle=True)
        data = file[file.files[0]].item()
        burn = st.slider('burn', 10,10000,400)

    # loss plot

    cols=st.columns(2)
    fig = plt.figure()
    plt.title('Loss')
    plt.xlabel('iter')
    _ = [plt.plot(data[i]['losses'], c='white', alpha=.3) for i in data]

    plt.axvline(burn, ls='--', c='r', label='burn')
    #plt.ylim(0,1e4)
    plt.close()
    cols[0].pyplot(fig)


    # param hist2d plot
    ms = np.array([data[i]['ms'][:,:,:] for i in data])#.shape
    ms = ms[:,burn:, :,:]
    ms = ms.reshape(ms.shape[0]* ms.shape[1], 2, 5)


    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(3, 2, figure=fig)
    ax = [fig.add_subplot(gs[0, :]), 
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), 
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    for i in range(5):
        ax[i].hist2d(ms[:,0,i], ms[:,1,i], bins=50);

    plt.tight_layout()
    plt.close()
    cols[1].pyplot(fig)


def sphereINcube_demo(data = []):
    return None
    # guess
    fig_guess, ax_guess = plt.subplots(figsize=(4,2))

    n_dims = np.arange(2,10)
    p = np.pi/2**n_dims
    cols[1].markdown('p = np.pi/2**n_dims')
    plt.title('guess', color='white')
    plt.plot(n_dims, p, c='black', lw=2, ls='--', label="guess")
    plt.xlabel('number of dimensions', color='white')
    plt.ylabel(r'% inside unit hypersphere', color='white')
    #
    plt.legend()
    
    logscale = tog.st_toggle_switch(label="Log scale", 
                key="Key1", 
                default_value=False, 
                label_after = False, 
                inactive_color = '#D3D3D3', 
                active_color="#11567f", 
                track_color="#29B5E8"
                )
    cols = st.columns(2)
    #logscale = cols[1].radio('log scale?', [True, False])
    if logscale:plt.yscale('log')
    c = cols[1].empty()

    
    # accept, reject to get pi
    

    # inputs
    n_points = cols[0].select_slider('Number of points', np.logspace(1,14,14,base=2, dtype=int))
    
    
    n_dim = cols[0].select_slider('Number of dimensions', np.arange(2,10,1, dtype=int))
    cols[0].markdown('My guess might be slightly high?... Also, notice how in 3 dim. a ball (sort of) appeares.')
    p_norm = cols[0].slider('p (for norm)', 0.,8.,2.)
    cols[0].markdown(r"""
    $$
        ||x||_p = \left(\sum_i |x_i|^p\right)^{1/p}
    $$
    """)

    # make vecs and check norms
    X = np.random.uniform(-1,1,(n_points, n_dim))
    fig, ax = plt.subplots(figsize=(5,5))
    norm = np.sum(abs(X)**p_norm, axis=1)

    # plotting
    colors = [{0 : 'gold', 1 : 'green'}[n<=1] for n in norm]
    ax.scatter(X[:,0], X[:,1], c=colors,  norm = np.sum(X**2, axis=1)**.5, cmap='winter', alpha=.4)
    extent = 1.1 ; ax.set(xlim=(-extent,extent), ylim=(-extent,extent))
    
    # output
    cols[1].pyplot(fig)
    percentage = sum(abs(norm)<1)/n_points
    cols[0].write('Percentage inside the unit hypersphere = {:0.2f} giving us $\pi = {:0.4f}$'.format(percentage, percentage*4))
    
    data.append((n_dim, percentage))
    for d, perc in data:
        ax_guess.scatter(d, perc, label="data")
    ax_guess.legend()
    c.pyplot(fig_guess)
    
    cols[1].caption("we just show the first two dimensions, and the color indicates whether we are within the unit (hyper)sphere")