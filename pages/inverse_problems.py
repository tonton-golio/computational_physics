from utils.utils_inverse import *
import streamlit_toggle as tog
#st.title('Inverse Problems')

#"Course taught by: Klaus Mosegaard."

def landingPage():
    ''
    """# Inverse Problems"""
    text_dict = getText_prep(filename = text_path+'landingPage.md', split_level = 1)
    for key in text_dict:
        text_dict[key]

    """
    Inverse problems are problems in which the goal is to infer an unknown cause from its known effect.
    """

    cols = st.columns(3)
    cols[0].image('https://nci-media.cancer.gov/pdq/media/images/428431-571.jpg')
    cols[0].caption('magnetic resonance imaging (MRI)')

    cols[1].image('https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20170104105121/ri/590/picture/Brain_SPECT_with_Acetazolamide_Slices_thumb.jpg')
    cols[1].caption('single-photon emission computed tomography (SPECT')


    cols[2].image('http://www.earth.ox.ac.uk/~smachine/cgi/images/welcome_fig_tomo_depth.jpg')
    cols[2].caption('Seismic tomography')

    '''
    Inverse problems lets us *see* what's inside.
    '''

def informationTheory():
    ''
    r"""
    ## Information theory
    The suprise of an event is described by the information of that event, $I(E)$. The information of an event is given by the logarithm of the reciprocal probability;
    $$
    \begin{align*}
        \log\left(\frac{1}{p(E)}\right)
    \end{align*}
    $$
    So if we flip a coin, expecting heads, and getting tails, we are **1 bit** suprised. Entropgy describes the expected amount of surprisal:
    $$
    \begin{align*}
        H(p) &= \sum_i p(E_i)I(E_i)\\
        H    &= -\sum_i p_i\log_2(p_i)
    \end{align*}
    $$
    This is the **Shannon entropy**.
    

    By expanding to the continous domain, we obtain the definition of information in for probability densities:
    $$
        H(f) = -\int_{-\infty}^\infty f(x)\log_2(f(x))dx
    $$
    > We call it differential entropy. Differential entropy can take negative values and it is not necessarily a measure of the amount of information.

    > Relative entropy is translation invaritant

    ### Kullback Leibler divergence

    The Kullback-Leibler divergence (KL-divergence) is a measure of the difference between two probability distributions. 
    
    It is defined as the expected value of the logarithm of the ratio of the probability density functions of the distributions, with respect to the probability measure of one of the distributions. 
    $$
        D_\text{KL}(p||q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
    $$
    
    It is a non-negative value and it is zero if and only if the two distributions are identical. It is used to measure the amount of information lost when approximating one distribution with another.


    ### How do you choose a good model parameterization for inverse problems? 

    * Use physical insight to ensure that the parameters have clear physical interpretation and are consistent with the known physics of the problem.
    * Utilize sparsity and parsimony to improve computational efficiency and generalization ability by using models with a small number of parameters.
    * Incorporate prior information about the problem to guide the choice of model parameterization.
    * Use regularization to constrain the model parameterization and improve the stability of the solution.
    * Empirically validate the performance of the model parameterization by testing it on experimental or observational data.

    *It's worth noting that choosing a good model parameterization for inverse problems is often a trade-off between the accuracy and the computational complexity of the solution. It's essential to find a balance between the two.*
    """
    

def Probabilistic():
    ''
    r"""
    # Probabilistic formulation of inverse problems
    
    ### How are inverse problems formulated in a Bayesian setting? 

    In a Bayesian setting, an inverse problem is formulated as a probabilistic inference problem. The goal remains the inferencce of parameters given some observed data and a prior probability distribution over the parameters. The prior distribution encodes prior information about the parameters, such as physical constraints or previous measurements. 
    
    The likelihood function, which is a function of the observed data and the parameters, quantifies the probability of the data given the parameters. *We can consider this to be inverse loss.*
    

    Together, the prior and likelihood functions define the posterior distribution over the parameters, which is the target of the inference. This distribution can be calculated using Bayes' theorem,
    
    $$ 
        P(H|E) = \frac{P(E|H)P(H)}{P(E)}
    $$
    
    The posterior can be approximated using numerical methods such as Markov chain Monte Carlo (**MCMC**).

    ### Advantages
    * The Bayesian formulation provides a natural way to incorporate prior information or knowledge about the parameters of the system into the inference process. This allows for a more robust and efficient estimation of the parameters, especially when the data is noisy or incomplete.
    * It provides a probabilistic interpretation of the parameters, rather than point estimates. The posterior distribution represents our uncertainty about the parameters given the data, informing us of credible intervals.
    * Additionally, the Bayesian formulation can be used to model more complex and realistic systems, including systems with multiple parameters and non-linear relationships between the parameters and the data. Furthermore, it can also be used to model systems with uncertain or missing data, missing model components or with model uncertainty.
    * Finally, Bayesian methods allow to naturally incorporate uncertainty.

    ### Disadvantages
    * The Bayesian formulation can be computationally expensive, especially for high-dimensional or complex systems. The calculation of the posterior distribution typically requires solving high-dimensional integrals, which can be challenging or intractable for some problems. 
    > Markov Chain Monte Carlo (MCMC) methods, which are often used to approximate the posterior, can also be computationally intensive and may require a large number of iterations to converge.

    * Another weakness is that the choice of prior distribution is subjective, it may not always be clear how to choose a prior that accurately represents the uncertainty in the parameters, and different prior choices can lead to different posterior distributions and thus different conclusions.

    * Additionally, Bayesian methods can also be sensitive to the choice of the likelihood function, which determines how the data is compared to the model predictions, and it is not always straightforward how to select the best likelihood function for a given problem.

    * Finally, it may not always be clear how to decide between different models or hypotheses, as Bayesian methods can provide evidence for multiple models, even if one model is clearly better than the others.
    
    ### Tarantola-Valette Formalism
    The Tarantola-Valette formalism is a Bayesian approach to inverse problems that uses a probabilistic formulation to estimate the unknown parameters of a system. The general form of an inverse problem using this formalism can be represented as follows:

    Given a set of observed data, $d$, and a mathematical model, $g(m)$, that describes the relationship between the unknown parameters, m, and the data, the goal is to estimate the most likely values of $m$ given the data and some prior information about $m$.

    The likelihood function, which represents the probability of the data given the parameters, is defined as:

    $$
        P(d|m) = \frac{1}{(2\pi)^{n/2} |V|^{1/2}} exp(-\frac{1}{2}(d-g(m))^TV^{-1}(d-g(m))
    $$

    Where $V$ is the covariance matrix of the measurement errors and n is the number of data points.

    The prior information about the parameters is represented by a prior probability distribution, $P(m)$, which is assumed to be Gaussian with mean $\mu$ and covariance matrix $V$:

    $$
        P(m) = \frac{1}{(2\pi)^{n/2} |V|^{1/2}} exp(-\frac{1}{2}(m-\mu)^TV^{-1}(m-\mu))
    $$

    The posterior distribution, which represents the probability of the parameters given the data, is calculated using Bayes' theorem:

    $$
        P(m|d) \propto P(d|m)P(m)
    $$

    The most likely values of m can be estimated by finding the peak of the posterior distribution, which can be done using optimization algorithms or Markov Chain Monte Carlo (**MCMC**) methods.
    

    ### How does the Tarantola-Valette formalism differ from the Bayesian formulation? 

    The Tarantola-Valette formalism is a specific implementation of the Bayesian formulation for inverse problems. Both approaches use a probabilistic formulation to estimate the unknown parameters of a system, and they both rely on Bayes' theorem to calculate the posterior distribution. However, the Tarantola-Valette formalism assumes that the noise in the data is Gaussian and that the prior and likelihood are both Gaussian distributions.

    This assumption of Gaussian distributions may simplify the computations, but also it may not be always the case in practice. Additionally, the Tarantola-Valette formalism is a linearized Bayesian approach, it makes assumptions about the linearity of the model and the Gaussianity of the error, which might not hold in some situations.

    The Bayesian formulation is more general and flexible, and it can handle non-Gaussian noise and priors, non-linear models, and multiple hypotheses. This generality can make it more computationally intensive or harder to implement, but it is also more suitable for complex and realistic systems.

    In summary, the Tarantola-Valette formalism is a specific Bayesian approach that is based on Gaussian assumptions and linearization, while the Bayesian formulation is a more general approach that can handle a wider range of problems.
    """

def DensityVar_LeastSquare():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
    # Excercise
    st.markdown(text_dict['Ex 1'])

    ## read data
    path_data = 'assets/inverse_problems/data/gravdata.txt'
    arr = np.loadtxt(path_data)
    xs, d_obs = arr[:,0], arr[:,1]

    ## display data
    with st.expander('data', expanded=False):
        df = pd.DataFrame(arr.copy(), columns=['x','d_obs (1e9)'])
        df['d_obs (1e9)'] *= 1e9
        df.T

    ## Discretization
    st.markdown(text_dict['Ex 2'])

    ## calc and show G
    G = G_matrix(xs=xs, zs=np.arange(100))
    cols = st.columns(2)
    cols[0].markdown(text_dict['Ex 3'])
    cols[1].pyplot(contour_of_G(G.T))

    ## calc and show ms
    eps_space = np.logspace(-13, -9, 60)
    ms = getParams(G, d_obs, eps_space)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.contourf(ms,10, cmap=plt.cm.inferno)
    fig.set_facecolor('lightgray')
    tick_locs = np.arange(len(eps_space))[::len(eps_space)//10]
    tick_vals = np.round(eps_space,13)[::len(eps_space)//10]
    ax.set(yticks=tick_locs,yticklabels=tick_vals)
    ax.set_ylabel('epsilon', color='black')
    ax.set_xlabel('depth', color='black')

    plt.close()
    cols = st.columns(2)
    cols[0].pyplot(fig)
    cols[1].markdown(text_dict['Ex 4'])
    

    ## Find minimum 
    fig = find_minimum(G, ms, d_obs, eps_space,
                        data_error = [10**(-9)] * 18,
                        data_error1 = [10**(-8)] * 18)


    st.markdown(text_dict['Ex 5'])
    st.pyplot(fig)
    st.caption('Precision becomes as resolution is increased.')
    
def Least_squares():
    text_dict = getText_prep(filename = text_path+'Tikonov.md', split_level = 1)
    st.markdown(r'{}'.format(text_dict['1']))


def ass1():
    import numpy as np
    import matplotlib.font_manager
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    import streamlit as st
    from matplotlib.transforms import Bbox
    dark_color = (0,.1,.15)
    mpl.rcParams['patch.facecolor'] = dark_color
    mpl.rcParams['axes.facecolor'] = 'grey'
    mpl.rcParams['figure.facecolor'] = dark_color
    

    def getText_prep_1(filename = text_path+'linear_tomography.md', split_level = 1):
        """
            get text from markdown and puts in dict
        """
        # extra function
        with open(filename,'r', encoding='utf8') as f:
            file = f.read()
        level_topics = file.split('\n'+"#"*split_level+' ')
        text_dict = {i.split("\n")[0].replace("#"*split_level+' ','') : 
                    "\n".join(i.split("\n")[1:]) for i in level_topics}
        
        return text_dict  

    def coverImage(n_seismographs=20, b = {'left':1,'right':4,'bot':1,'top':5}):
        def makeGround(x, y ,b = b):
            Z =  np.zeros((len(x),len(y)))
            X, Y = np.meshgrid(x,y)
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    if (b['left']< xi < b['right']) and (b['bot'] < yj < b['top']):
                        Z[i,j] = 1
            m = Z.copy().flatten()

            dx = x[1]-x[0]
            dy = y[1]-y[0]
            dist = (dx**2+dy**2)**.5 * 1000 
            Z *= dist * ((1/5000) - (1/5200))
            #Z
            return Z, X, Y, m

        def traceRays(x, y, Z, seismograph_locs):
            n= len(seismograph_locs)
            x_fine = np.linspace(min(x),max(x), n+1)
            y_fine = np.linspace(min(y),max(y), n+1)

            def linear(x, a, b):
                return a*x + b
            rays = {}
            ray_values_left = np.zeros(n)
            ray_values_right = np.zeros(n)
            G_left = np.zeros((n, n, n))
            for idx, s_loc in enumerate(seismograph_locs):
                # coming from the left
                xi_left = x_fine[x_fine<s_loc]
                yi_left = linear(xi_left, -1,s_loc)
                #then we need to check which box each is in 

                ii_left = [np.argmin(abs(x-i)) for i in xi_left]
                jj_left = [np.argmin(abs(y-j)) for j in yi_left]
                #ii_left
                #G_left[s_loc]
                #a, b  = np.argwhere(x_fine<s_loc), np.argwhere(x_fine<s_loc)
                #a
                ray_values_left[idx] = sum([Z[i,j] for i, j in zip(ii_left, jj_left)])
        
                # coming from the right
                xi_right = x_fine[x_fine>s_loc]
                yi_right = linear(xi_right, 1,-s_loc)

                ii_right = [np.argmin(abs(x-i)) for i in xi_right]
                jj_right = [np.argmin(abs(y-j)) for j in yi_right]
                ray_values_right[idx] = sum([Z[i,j] for i, j in zip(ii_right, jj_right)])
                
                rays[s_loc] = (xi_left, yi_left, xi_right, yi_right)

            #G_left
            ray_values = np.concatenate([ray_values_left, ray_values_right])
            return rays, ray_values_left, ray_values_right , ray_values

        def plot():
            def plot_rays(rays, ax=plt):
                for i in rays:
                    (xi_l, yi_l, xi_r, yi_r) = rays[i]

                    ax.plot(xi_l, yi_l, c="white", lw=0.75)
                    ax.plot(xi_r, yi_r, c="white", lw=0.75)

            fig, ax = plt.subplots(2,1)
            fig = plt.figure(constrained_layout=True)

            gs = GridSpec(3, 1, figure=fig)
            ax = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[2])]
            ax[0].scatter(X, Y, c=Z.T, marker=',', cmap='winter', s=(400/n_seismographs)**2, alpha=.7)

            plot_rays(rays, ax[0])
            ax[0].set_ylim(ax[0].get_ylim()[::-1])
            
            #for axi in ax:
                #axi.set(facecolor=dark_color)

            ax[0].set_xlabel('x', color='white',
            size=14)
            ax[0].set_ylabel('depth', color='white',
            size=14)
            
            
            ax[1].set_xlabel('Siesmograph index', color='white', 
            size=14)
            ax[1].set_ylabel('$t_\gamma$', color='white', 
            size=14)
            ax[1].bar(np.arange(len(d_left))-0.35, d_left,width=.25)
            ax[1].bar(np.arange(len(d_right))+0.35, d_right,width=.25)
            #fig.set_facecolor(dark_color)
            plt.tight_layout()
            return fig

        x = np.linspace(0,14, n_seismographs)
        y = np.linspace(0,12, n_seismographs)

        Z, X, Y, m = makeGround(x,y)
        seismograph_locs = np.linspace(1,max(x)-1, n_seismographs)

        rays, d_left, d_right, d_obs = traceRays(x, y, Z, seismograph_locs)
        
        fig = plot()
        return d_obs, fig, m,

    def make_G(N=13):
        G_right = [np.eye(N,k=1+i).flatten() for i in range(N-2)]
        G_left = [np.flip(np.eye(N,k=-(1+i)), axis=0).flatten() for i in range(N-2)]
        
        z = np.zeros((1,N**2))
        G = np.concatenate([z, G_left[::-1],z, z, G_right,z])

        G *= 2**.5 * 1000
        return G

    def plot_G_summed(G,N):
        G = G.copy()/ (2**.5 * 1000)
        fig, ax = plt.subplots( figsize=(6,3))
        
        plt.imshow(G.sum(axis=0).reshape(N,N), extent=(0,N,N,0))
        plt.xticks([0,N//2,N])
        plt.colorbar()
        return fig

    def make_m(top=2, bot=8, left=3, right=7):
        '''the section of earth under investigation'''
        M = np.zeros((N,N))
        M[top:bot, left:right] = (1/5000) - (1/5200)
        return M.flatten()

    def forward(G, m, noise_scale=1/18,seed=42):
        'returns d'
        def addNoise(t_pure, seed):
            np.random.seed(seed)
            noise = np.random.randn(len(G))
            noise /= np.linalg.norm(noise)
            n = noise * noise_scale * np.linalg.norm(t_pure)
            t_obs = t_pure + n
            return t_obs, np.linalg.norm(n)
        
        d_pure = G@m
        d_obs, n_norm = addNoise(d_pure, seed)
        return d_obs, n_norm

    def genDataPlot(m_true, d_obs, st=st):
        fig, ax = plt.subplots(1, 3, figsize=(12,3))

        ax[0].imshow(G, aspect=6)
        ax[0].set_title('$G$', color='white')
        ax[1].imshow(m_true.reshape(N,N), extent=(0,N,N,0))
        ax[1].set_title('$m$', color='white')  

        ax[2].set_title(r'$d_{obs}$', color='white')
        ax[2].bar(np.arange(N)-0.35, d_obs[:N],width=.35 , label='left')
        ax[2].bar(np.arange(N)+0.35, d_obs[N:],width=.5, label='ight')
        ax[2].legend()
        ax[2].set_xticks(*[range(N)[::4]]*2)
        ax[2].set_xlabel('Detector number', color='white')
        st.pyplot(fig)

    def backward(d_obs, n_norm, G, n_eps=10):
        epss, offs, ms = np.logspace(-3, 4, n_eps), [], []

        for eps in epss:
            m = np.linalg.inv(G.T@G  + eps**2*np.eye(N**2)) @ (G.T @ d_obs) #least square
            off = np.abs( np.linalg.norm(G@m - d_obs) - n_norm) # residual
            offs.append(off);  ms.append(m)

        
        epss_zoom = np.linspace(epss[np.argmin(offs)-1],
                                    epss[np.argmin(offs)+1],
                                    n_eps*2)
        offs_zoom, ms = [], []
        for eps in epss_zoom.copy():
            m = np.linalg.inv(G.T@G  + eps**2*np.eye(N**2)) @ (G.T @ d_obs) #least square
            off = np.abs( np.linalg.norm(G@m - d_obs) - n_norm) # residual
            offs_zoom.append(off);  ms.append(m)
        m_opt = ms[np.argmin(offs_zoom)]
        return m_opt, epss, offs, epss_zoom, offs_zoom

    def plot_m_opt(m_opt):
        fig, ax = plt.subplots()
    

        
        ax.imshow(m_opt.reshape(N,N), extent=(0,N,N,0))
        
        ax.set(xticks=np.arange(2,14,2)-.5, xticklabels=range(2,14,2))
        ax.set_title('Pred. density of earth', color='white')
        ax.set_xlabel('x', color='white')

        return fig

    def plot_epss(epss, offs, epss_zoom, offs_zoom):
        fig, ax1 = plt.subplots()
        #T_E = np.arange(1,max(T)+1,1)
        # The data.
        ax1.plot(epss, offs, c='white', lw=4)
        ax1.set_xlabel(r'$T\,/\mathrm{K}$')
        ax1.set_ylabel(r'$C_p\,/\mathrm{J\,K^{-1}\,mol^{-1}}$')
        
        ax1.set(yscale = 'log', xscale = 'log',facecolor=dark_color)
    
        ax1.set_xlabel('epsilon', color='white', fontsize=14)
        ax1.set_ylabel('|t_obs - G@m|', color='white', fontsize=14)
        
        ax2 = fig.add_axes((0.3,.6,0.4,.3))
        
        ax2.set(yscale = 'log', xscale = 'log')#,facecolor=dark_color)
        ax2.set_facecolor('white')
        ax2.plot(epss_zoom, offs_zoom, c=dark_color)

        return fig



    # main render
    text_dict = getText_prep_1()

    "# Linear Tomography"
    st.markdown(text_dict['intro'])

    def sliders_5():
        cols = st.columns(5)
        top = cols[0].slider('top', 0., 11., 3.18)
        bot = cols[1].slider('bot', 0., 11., 4.5)
        left = cols[2].slider('left', 0., 11., 9.)
        right = cols[3].slider('right', 0., 13., 13.)
        n_seismographs = cols[4].select_slider('n seis.', range(12,99,20))
        return top, bot, left, right, n_seismographs
    top, bot, left, right, n_seismographs = sliders_5()

    d, fig, m = coverImage(n_seismographs, b = {'left':min(left,right), 'right':max(left,right), 'bot':min(bot,top), 'top':max(bot,top)})
    st.pyplot(fig)



    ## G
    '### Generating data'
    cols=st.columns(2)
    cols[0].markdown(text_dict['Generating data'])

    st.code(text_dict['G code'])
    cols_3=st.columns(3)

    N = cols_3[2].slider('N', 1,50,12)
    G = make_G(N)
    fig = plot_G_summed(G, N); cols[1].pyplot(fig)

    top = cols_3[0].slider('top', 0, N, 4)
    bot = cols_3[0].slider('bot', 0, N, 5)
    left = cols_3[1].slider('left', 0, N, 3)
    right = cols_3[1].slider('right', 0, N, 5)
    m_true = make_m(min(top, bot), max(top, bot), min(left, right), max(left, right))

    noise_scale = 1 / cols_3[2].slider('1/Noise_scale', 1,50,18)

    d_obs, n_norm = forward(G, m_true, noise_scale, 111)
    genDataPlot(m_true, d_obs, st=st)


    ## predicting m
    '### Predicting $m$'
    cols = st.columns(2)
    cols[0].markdown(text_dict['Predicting m'])

    n_eps = cols[1].slider('n_eps', 2, 99, 10)
    '**This yields the prediction of the internal structure of the earth shown below.**'
    m_opt, epss, offs, epss_zoom, offs_zoom = backward(d_obs, n_norm, G, n_eps)
    # maybe add an insert
    fig = plot_m_opt( m_opt,)

    fig2 = plot_epss( epss, offs, epss_zoom, offs_zoom)
    cols[1].pyplot(fig2)
    st.pyplot(fig)

    st.markdown(text_dict['A delta function'])

def monteCarlo():
    ''
    def old():
        text_dict = getText_prep(filename = text_path+'week4.md', split_level = 1)

        st.title('Non linearities')
        cols = st.columns(2)
        st.markdown(text_dict['Header 1'])
        st.markdown(text_dict['Header 2'])

    """
    # Monte Carlo
    ### For which type of inverse problems are Monte Carlo (MC) methods used? 
    Monte Carlo (MC) methods are often used for inverse problems that involve complex or high-dimensional systems, where the likelihood function and/or the prior distribution cannot be easily calculated or integrated analytically. These methods involve generating a large number of random samples from the prior distribution, and then evaluating the likelihood function for each sample. The samples that have a high likelihood are considered to be more probable, and they are used to estimate the posterior distribution.

    Markov Chain Monte Carlo (MCMC) methods are a class of MC methods that are particularly useful for inverse problems. These methods involve generating a sequence of samples that are correlated with each other, and they are designed to converge to the true posterior distribution.

    MCMC methods are often used for inverse problems that involve high-dimensional or non-linear systems, where the posterior distribution is not trivial to calculate. These methods can be used to approximate the posterior distribution, and thus estimate the unknown parameters of the system. Additionally, these methods can also be used to calculate various quantities of interest, such as credible intervals or the expected value of a function of the parameters.

    MCMC methods are also used in a wide range of applications such as Bayesian statistics, physics, engineering, image processing, signal processing, and machine learning.
    """
    

# Navigator
topic_dict = {
    "Landing Page" : landingPage,
    'Information theory' : informationTheory,
    'Probabilistic inference' : Probabilistic,
    #'Tikonov': Least_squares, 
    #'Monte Carlo': monteCarlo,

    'Density variations (Tikonov)': DensityVar_LeastSquare,
    'Linear Tomography (Tikonov)' : ass1,
    'Vertical Fault (Monte Carlo)': ass2,
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()

