import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
from time import sleep
import matplotlib as mpl
from scipy.constants import gravitational_constant
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from numba import prange, jit

#import cv2
from time import time
from scipy.optimize import curve_fit

def gauss_pdf_N(x, mu, sigma):
    """Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

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
	mpl.rcParams['text.color'] =  "lightgreen" 
	mpl.rcParams['axes.labelcolor'] =  "lightgreen"
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

def ass3_glacier_thickness():
    #plt.style.use('bmh')

    # functions
    def init_progress_bar():
        progress_cols=st.columns(2)
        my_bar = progress_cols[0].progress(0.0)
        start = time()
        progress_info = progress_cols[1].empty()
        progress_info.write(f"Elapsed time = {0},  itr = {0} / {n_init},  ETA = {'?'}")
        return progress_cols, progress_info, my_bar, start

    def update_progress_bar(itr, n_itr, start, my_bar, progress_info):
        my_bar.progress(itr/n_itr)
        run_time = time()-start
        time_per_run = 1 if itr ==0 else run_time/itr
        progress_info.write(f"Elapsed time = {round(run_time,1)}, itr = {itr} / {n_itr}, ETA = {round(time_per_run*(n_itr-itr),2)}")

    def load_data():
        data = np.array([[535, -15.1,],
                        [749, -23.9,],
                        [963, -31.2,],
                        [1177, -36.9,],
                        [1391, -40.8,],
                        [1605, -42.8,],
                        [1819, -42.5,],
                        [2033, -40.7,],
                        [2247, -37.1,],
                        [2461, -31.5,],
                        [2675, -21.9,],
                        [2889, -12.9]])
        d_obs = data[:,1]
        x = data[:,0]
        dx = np.mean(x[1:]-x[:-1])
        return x, d_obs, dx

    def show_data(x, d):

        fig, ax = plt.subplots(1,1, figsize=(6,2))
        ax.scatter(x,d, label='observation', )
        ax.set(xlabel='distance from west edge (m)',
                ylabel = 'Bouguer anomaly (mGal)',
                xlim=(0,3.42*1e3), 
                title='Observations')
        #ax.legend(loc='upper center')
        plt.close()
        return fig

    def show_heights(h,x, dx, label='Toy heights'):
        fig, ax = plt.subplots(1,1, figsize=(5,3))
        ax.bar(x, height=h, width=dx, label=label)
        ax.legend()
        ax.set(xlabel='distance from west edge (m)',
                ylabel = 'glacier height',
                xlim=(0,3.42*1e3))
        plt.tight_layout()
        plt.close()
        return fig

    def gauss_pdf(x, mu, sigma, A):
        """Gaussian"""
        return A / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

    def obtain_estimate(h, eps):
        return np.array([G * del_rho * dxl * np.sum(np.log(((xl - xj)**2 + h**2) / ((xl - xj)**2+eps))) for xj in x]) * 10e5

    def loss(h, d_obs, d_est): 
        MSE = sum((d_obs -  d_est)**2)
        dh = abs(h[1:]-h[:-1])
        dx = 3420/len(h)
        dhdx = dh/dx
        height_loss = np.sum(dhdx**2)#sum(dh>1.01)/len(h)

        #st.write(dhdx, MSE, height_loss)
        return  MSE + height_loss*11.1

    def log_likelihood(h, d_obs, d_est):
        ''
        C = np.eye(len(d_obs)) #* np.random.normal(10,1, len(d_obs))
        #C /= np.linalg.norm(C)
        C *= 0.0012**2

        vector_product = (d_obs-d_est).T @ np.linalg.inv(C) @ (d_obs-d_est)
        constant=1
        return np.log(constant) -.5 * vector_product


    def log_rho(m, m0):
            constant = 1
            C_m = np.eye(len(m))
            #C_m /= np.linalg.norm(C_m)
            C_m *= 300**2
            result =np.log(constant)  -.5 * (m-m0).T @ np.linalg.inv(C_m) @ (m-m0)
            #'log_rho', result 
            return result 

    def sigma(m, m0, d_obs, d_est):
            #m.shape
            #m0.shape
            #d_obs.shape
            #d_est.shape
            constant = 1
            likelihood = log_likelihood(m, d_obs, d_est)
            #'log likelihood', likelihood
            return np.log(constant)  + log_rho(m, m0) -1*likelihood
        

    def evaluate(x, h, d_obs, eps):
        d_est = obtain_estimate(h, eps)
        fig, ax = plt.subplots(1,1, figsize=(5,3))
        ax2 = ax#.twinx()
        ax.scatter(x,d_obs, label='observation')
        ax2.scatter(x,d_est, label='estimate', marker='x', c='r')
        likelihood_ = log_likelihood(h, d_est, d_obs)
        ax.set(xlabel='distance from west edge (m)',
                ylabel = 'Bouguer anomaly (mGal)',
                title=f'likelihood = {likelihood_}',
                xlim=(0,3.42*1e3))
        ax.legend(loc='upper center')
        ax2.legend(loc='upper center')
        plt.close()
        return fig

    def MCMC(h0, eps=100, n_itr=5, lr=1, beta=1):
        
        d = obtain_estimate(h0, eps)
        likelihood_ = log_likelihood(h0, d,  d_obs)
        h = h0.copy()
        
        hs, likelihoods, n_accepted = [h.copy()], [likelihood_], 0
        for i in range(n_itr):
            h_alt = abs(np.random.normal(h, lr))
            d_alt = obtain_estimate(h_alt, eps)
            likelihood_alt = log_likelihood(h_alt, d_alt,  d_obs)
            
            delta_likelihood = likelihood_alt-likelihood_
            if delta_likelihood > 0:
                # accept change
                n_accepted +=1
                d, h, likelihood_ =d_alt.copy(), h_alt.copy(), likelihood_alt.copy() 
            else:
                
                p = np.exp(delta_likelihood*beta)
                #print(p)
                r = np.random.uniform(0,1)
                if r<p:
                    #print("accept")
                    n_accepted +=1
                    d, h, likelihood_ =d_alt.copy(), h_alt.copy(), likelihood_alt.copy()
            
            hs.append(h.copy())
            likelihoods.append(likelihood_)
        return np.array(hs), np.array(likelihoods), n_accepted/n_itr

    def MCMC_multi(h0, ninit=10, nitr=1000, burn=300, lr=1, start_jump=1, verbose=True, beta=2):
        # multi MCMC
        hs = []
        likelihoods = []
        acc_ratios = []

        progress_cols, progress_info, my_bar, start = init_progress_bar()

        for init in prange(n_init):
            if verbose: 
                update_progress_bar(init, n_init, start, my_bar, progress_info)

            h0_ = abs(np.random.normal(h0, start_jump))

            hs_, likelihoods_, acc_ratio = MCMC(h0_, epsilon, nitr, lr=lr, beta=beta)

            hs.append(hs_)
            likelihoods.append(likelihoods_)
            acc_ratios.append(acc_ratio)
        update_progress_bar(init+1, n_init, start, my_bar, progress_info)
        hs = np.vstack([h[burn:] for h in hs])
        likelihoods =np.vstack(likelihoods)
        return hs, likelihoods, np.array(acc_ratios)

    def show_likelihoods(likelihoods, burn):
        fig, ax = plt.subplots(figsize=(5,3))
        for L in likelihoods:
            ax.plot(-1*L)
        ax.axvline(burn, c='black', ls='--')

        ax.set(xlabel='itr', ylabel='- log(likelihood)', 
                yscale='log', #xscale='log'
                #ylim=(min(L), 10**4)
                )
        
        ins = ax.inset_axes([0.55,0.55,0.35,0.35])
        ins.set_xticks([np.mean(acc_ratios)], c='black')
        ins.boxplot(acc_ratios, vert=False)
        ins.grid()
        #ins.title('Boxplot of the acceptance ratio from initializations')
        


        plt.close()
        return fig

    def plot_acceptance_ratios(acc_ratios):
        fig = plt.figure(figsize=(5,3))
        plt.boxplot(acc_ratios)
        plt.grid()
        plt.title('Boxplot of the acceptance ratio from initializations')
        plt.close()
        return fig

    # data
    x, d_obs, dx = load_data()

    #d_obs *= 1e-5
    G = gravitational_constant
    del_rho = -1733
    a = 3.42*1000
    img = plt.imread('assets/inverse_problems/images/glacier_valley_diagram.png')


    # intro
    st.title('Glacier thickness')
    cols = st.columns(2)
    cols[0].image(img)
    cols[1].markdown("""
    A glacier fills a valley. We want to know how thick it is. We measure the pull of gravity at 12 points as marked by the little stars. """)
    cols[1].markdown("""
    Since ice has lower density than rock, we observe a greater gravity-anomaly where there is a thicker layer of ice. 
    """)


    # equations
    st.markdown(r"""
    #### Governing equation
    So we have this equation, which tells us what we should observe given we know the height of the ice.

    $$
        \Delta g_j = \Delta g(x_j) =
        G\Delta \rho \int_0^a \ln\left[\frac{(x-x_j)^2 + h(x)^2}{(x-x_j)^2}\right]dx
    $$
    This has a singularity which can be disregarded as it is not physically permissible. I offset the numerator by quite a bit as not to have great relative-size skew. In discrete form we have: 

    $$
    \Delta g(x_j) \approx
        G\Delta \rho 
        \sum_l 
        \ln
        \left[\frac{(x_l-x_j)^2 + h(x_l)^2}{(x_l-x_j)^2+\varepsilon}\right] \Delta x
    $$
    """)



    # Initial parameter guess
    r"""
    #### Initial parameter guess
    To come up with a starting point for our Monte Carlo search,, I'll assume that the height of ice is great near the middle and tapers to either side. This is well described by a Gaussian PDF.

    To obtain starting parameter values, we need to know how good a set of parameters is, we know this from the likelihood.

    $$
        L(\mathbf{m}) = k \cdot \exp\left(
            -\frac{1}{2} (\mathbf{d}_\text{obs}-g(\mathbf{m}))^T \mathbf{C}_d^{-1} (\mathbf{d}_\text{obs}-g(\mathbf{m}))
            \right)
    $$

    With that in place, I'll adjust sliders to try and make for a sensible starting point.

    """

    cols=st.columns(2)
    cols[1].pyplot(show_data(x,d_obs))
    space = cols[0].radio('space', ['linspace', 'arange'])
    if space == 'linspace':
        size = cols[0].slider(   'size',    4, 100,     10)
        xl = np.linspace(0, a, size) ; dxl = xl[1]-xl[0] ; 
        first_right = (x[0]-xl)[x[0]-xl < 0][0]
        last_left = (x[-1]-xl)[x[-1]-xl > 0][-1]
        first_right, last_left
        shift = first_right+last_left
        xl-=shift
        shift
    else:
        space = cols[0].slider('space = dx/',    1, 16,     4)
        xl = np.arange(0,a, dx/space)
        dxl = xl[1]-xl[0] ; 
        size = len(xl)
        cols[0].write(f'size = {size}')
        xl+=dxl/2

    fig, ax = plt.subplots(2,1)
    count = [len(xl[ (xl>x[i]) & (xl<x[i+1])]) for i, _ in enumerate(x[:-1])]
    count = np.array(count).reshape(1,-1)
    ax[0].set_title('number of $x$-slices between detectectors')

    im = ax[0].imshow(count, cmap='RdBu')
    fig.colorbar(im, ax=ax[0],aspect=.6)
    #ax[0].colorbar(aspect=.6)
    ax[0].set_yticks([])
    ax[0].set_xticks([-.5+i for i in range(12)], range(1,13))


    d = [min(abs(xi-xl)) for i, xi in enumerate(x)]
    d = np.array(d).reshape(1,-1)
    plt.title('distance to nearest $x$-slice')

    plt.imshow(d, cmap='RdBu')
    plt.colorbar(aspect=.6)
    plt.yticks([])
    plt.xticks([i for i in range(12)], range(1,13))


    plt.tight_layout()
    plt.close()
    cols[1].pyplot(fig)




    if space == 'linspace':
        epsilon = cols[0].slider('epsilon', 0, int(dx), int(dxl))
    else:
        epsilon = 0
    mu = cols[0].slider(     'mu',      0, int(a),  int(x[0]+x[-1])//2)
    scale = cols[0].slider(   'scale',  1, 1000,    920)
    A = cols[0].slider('Amplitude',     1, 100,     74)

    h0 = gauss_pdf(xl, mu, scale, 1)
    h0 /= max(h0) 
    h0 *= A


    m0  = d_obs / del_rho / G / 2 / np.pi

    #if size==16:
    #h0 = np.concatenate([np.ones(2), m0, np.ones(2)])*1e-6

    cols[1].pyplot(show_heights(h0, xl, dxl, 'initial guess'))
    cols[1].pyplot(evaluate(x, h0, d_obs, epsilon))

    cols[0].markdown("""
    So we now have a reasonable place to start!
    """)




    # MCMC
    '#### MCMC'

    cols = st.columns(6)
    n_init = cols[0].slider('no. initialization', 1,10,2)
    n_itr = cols[1].slider('no. iterations', 0,5000,2000, 100, )
    burn = cols[2].slider('burn-in', 0,n_itr,n_itr//2, 100)
    lr = cols[3].select_slider('learning rate', np.round(np.logspace(-4, 3, 8), 5))
    beta = cols[5].select_slider('beta', np.round(np.logspace(-10, -1, 10), 10))


    start_jump = cols[4].slider('start jump', 0.,5.,0., 0.1)

    if st.button('run'):

        hs, likelihoods, acc_ratios = MCMC_multi(h0, n_init, n_itr, burn, lr, start_jump, beta=beta)

        h = hs[-1]
        hs

        cols = st.columns(2)
        cols[0].pyplot(show_likelihoods(likelihoods, burn))
        cols[1].pyplot(evaluate(x, h, d_obs, epsilon))
        
        
        #cols[2].pyplot(plot_acceptance_ratios(acc_ratios))

        

        fig,ax  = plt.subplots()
        plt.scatter(xl, np.mean(hs, axis=0), marker='.')
        plt.errorbar(xl, np.mean(hs, axis=0), np.std(hs, axis=0),
                    lw=0, elinewidth=1)

        ax.set(xlabel='$x$ position', ylabel='glacier height', title='prediction')
        plt.close()
        st.pyplot(fig)

        

        def obtain_estimate2(h, eps, x):
            return np.array([G * del_rho * dxl * np.sum(np.log(((xl - xj)**2 + h**2) / ((xl - xj)**2+eps))) for xj in x]) * 10e5

        sum_sigs = 0 
        for h in hs:
            sig = sigma(h, h0, d_obs, obtain_estimate(h, epsilon))

            sig2 = sigma(h, h0, d_obs[::2], obtain_estimate2(h, epsilon, x[::2]))

            sum_sigs += np.log2(sig/sig2) * sig
        sum_sigs /= len(hs)
        sum_sigs
    if st.button('many'):

        fig, ax = plt.subplots(len(hs.T)//4+1,4, figsize=(16,2*len(hs.T)//4), sharey=True)
        fig.suptitle('Distributions of all parameters', fontsize=24)
        for i, m in enumerate(hs.T):
            try:
                counts, bins = np.histogram(m, bins=24)
                ax[i//4, i%4].stairs(counts, bins)

                x = (bins[1:] + bins[:-1])/2
                y= counts
                x_plot = np.linspace(min(x), max(x), 100)

                # fit all of these
                p0 = [30,10,1e5]
                popt, pcov = curve_fit(gauss_pdf, x, y,p0=p0)
                ax[i//4, i%4].plot(x_plot, gauss_pdf(x_plot, *popt))
            except: 
                counts, bins = np.histogram(m, bins=24)
                ax[i//4, i%4].stairs(counts, bins)
        plt.close()
        st.pyplot(fig)

        # two neighbouring parametes

        fig = plt.figure()
        plt.hist2d(hs.T[8],hs.T[9], bins=24, cmap='RdBu')
        plt.xlabel('height 9')
        plt.ylabel('height 10')
        plt.grid()
        plt.close()
        st.pyplot(fig)

    

def sphereINcube_demo(data = []):
    #return None
    # guess
    fig_guess, ax_guess = plt.subplots(figsize=(4,2))

    n_dims = np.arange(2,10)
    cols = st.columns(2)
    p = np.pi/2**n_dims
    cols[1].markdown('p = np.pi/2**n_dims')
    plt.title('guess', color='white')
    plt.plot(n_dims, p, c='black', lw=2, ls='--', label="guess")
    plt.xlabel('number of dimensions', color='white')
    plt.ylabel(r'% inside unit hypersphere', color='white')
    #
    plt.legend()
    
    #logscale = tog.st_toggle_switch(label="Log scale", 
    #            key="Key1", 
    #            default_value=False, 
    #            label_after = False, 
    #            inactive_color = '#D3D3D3', 
    #            active_color="#11567f", 
    #            track_color="#29B5E8"
    #            )
    logscale = True
    
    #logscale = cols[1].radio('log scale?', [True, False])
    if logscale:plt.yscale('log')
    c = cols[1].empty()

    
    # accept, reject to get pi
    

    # inputs
    n_points = cols[0].select_slider('Number of points', np.logspace(1,14,14,base=2, dtype=int))
    
    
    n_dim = cols[0].select_slider('Number of dimensions', np.arange(2,11,1, dtype=int))
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
    cols[0].write('Percentage inside the unit hypersphere = {:0.4f} giving us $\pi = {:0.4f}$'.format(percentage, percentage*4))
    
    data.append((n_dim, percentage))
    for d, perc in data:
        ax_guess.scatter(d, perc, label="data")
    ax_guess.legend()
    c.pyplot(fig_guess)
    
    cols[1].caption("we just show the first two dimensions, and the color indicates whether we are within the unit (hyper)sphere")