from streamlit_profiler import Profiler
p = Profiler()
p.start()

from utils.utils_inverse import *
import streamlit_toggle as tog
from scipy.special import rel_entr
import string
#st.title('Inverse Problems')

#"Course taught by: Klaus Mosegaard."
set_rcParams()

def strip_leading_spaces(x):
    x_stripped = x
    if len(x)<2: return x
    for i in range(12):
        try:
            if x_stripped[0] == ' ':
                x_stripped = x_stripped[1:]
            else:
                break
        except:
            break
    return x_stripped

def strip_lines(text):
    return'\n'.join([strip_leading_spaces(x) for x in text.split('\n')])

#st.write(strip_leading_spaces('    asdasd'))
def wrapfig(width=200, text='aaa',src='', st=st):
        
    
    HEAD = """<!DOCTYPE html>
        <html>    
        <head><style>
            img {
            float: right;
            margin: 5px;
            }
        </style></head>
        """
    
    BODY = """
        <body>
        <div class="square">
            <div>
            <img src="{}"
                width = {}
                alt="Longtail boat in Thailand">
            </div>
        <p>{}</p>
        </div>
        </body></html>
        """.format(src, width, text)

    str = HEAD+BODY
    str = strip_lines(str)
    st.markdown(str,  unsafe_allow_html=True
    )
    

def entropy_discrete(x):

    H  = 0
    for i in set(x):
        p = len(x[x==i])/len(x)
        H += p*np.log2(p)

    return -1*H

def entropy_continous(f, x):
    
    dx = x[1]-x[0]
    fx = f(x)
    fx = fx[fx>0]
    H = -1*sum(fx*np.log2(fx)*dx)
    return H


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
    cols = st.columns(2)
    cols[0].title('Information theory')
    
    # intro 
    cols[0].write(r"""
    Entropy and its relevance in data science: 
    > answers the question: *how similar are two samples?*
    """)
    cols[1].image('https://d2r55xnwy6nx47.cloudfront.net/uploads/2020/12/Claude-Shannon_2880_Lede.jpg', 
            width=300, caption='Claude Shannon, the "father of information theory". Shannon demonstated in his thesis, that boolean algebra can construct any logical numerical relationship.')

    '---'

    st.write(r"""
    The suprise of an event is described by the information of that event. *A single coinflip yield **1 bit** of suprise.*
    $$
         I(E) = \log_2\left(\frac{1}{p(E)}\right) = -\log_2(p(E)),
    $$
    in which $p(E)$ is the probability of an event $E$.
    """)
    st.caption('We typically use base-2 in CS. It tells us the number of yes/no questions we must answer to for full exploration.')

    st.write(r"""
    **Entropy** is the expected amount of surprise. **Shannon entropy** defines this quantoty in discrete-space;
    $$
        H_s = -\sum_i p(E_i)I(E_i).
    $$
    In the continous space, we define entropy for probability-densities,
    $$
        H_d(f) = -\int_{-\infty}^\infty f(x)\log_2(f(x))dx.
    $$
    > We call it differential entropy. Differential entropy can take negative values and it is not necessarily a measure of the amount of information. Note; relative entropy is translation invaritant.
    ---
    """)


    def discrete_continous_entropy_DEMO():

        cols = st.columns(2)
        
        cols[0].markdown(
        r"""
        ###### Comparing discrete and continous entropy
        
        """)


       
        
        def sliders(st=st):
            loc = st.slider('loc', -20,4,-8)
            scale =st.slider('scale', 0.1,10.,1., 0.1)
            size = st.select_slider('size', 2**np.arange(4,11))

            return loc, scale, size

        
        loc, scale, size = sliders(cols[0])
        
        # Calculations
        ## discrete
        x_discrete = np.random.normal(loc, scale, size).astype(int)
        H_discrete = entropy_discrete(x_discrete)

        ## continous
        f = lambda x: gauss_pdf_N(x, loc, scale)
        x_cont = np.linspace(-100, 100, size)
        H_cont = entropy_continous(f, x_cont)

        # plotting
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(4,4))
        fig.suptitle('Entropy of discrete and cont. distributions')
        counts, bins, _ = ax[0].hist(x_discrete, bins=range(min(x_discrete), max(x_discrete)+1), color='red', alpha=.8)
        #ax[0].set(xlabel='value', ylabel='occurence freq.')
        ax[0].text(7, max(counts)*.8, f'$H_s$ = {round(H_discrete,3)}', color='black', fontsize=16)

        ax[1].fill_between(x_cont, 0,f(x_cont), color='purple')
        ax[1].set(xlabel='value',)# ylabel='occurence freq.')
        ax[1].text(7, max(f(x_cont))*.8, f'$H_d$ = {round(H_cont,3)}', color='black', fontsize=16)
        plt.xlim(-40,40)
        plt.close()
        cols[1].pyplot(fig)
    
    discrete_continous_entropy_DEMO()

    st.markdown('---')
    # intro KL
    r"""
    ###### Kullback-Leibler divergence (**KL-divergence**)
    A measure of the difference between two probability distributions. 
    
    It is defined as the expected value of the logarithm of the ratio of the probability density functions of the distributions, with respect to the probability measure of one of the distributions. 
    $$
        D_\text{KL}(p||q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = -\sum_x p(x)\log\frac{q(x)}{p(x)}
    $$
    
    It is a non-negative value and it is zero iff the two distributions are identical. It is used to measure the amount of information lost when approximating one distribution with another.

    """
    '###### Aggression trait propensity'
    cols = st.columns(3)
    # left we control two distributions
    
    def sliders_and_data(st0=st, st1=st):
        loc1 = st0.slider('Average (male)', 0,10,5,1)
        scale1 = st0.slider('Devitation (male)', 0.1,10.,1.,.1)
        loc2 = st1.slider('Average (female)', 0,10,3,1)
        scale2 = st1.slider('Deviation (female)', 0.1,10.,1.0,.1)

        size = 400

        x1 = np.random.normal(loc1, scale1, size)
        x2 = np.random.normal(loc2, scale2, size)

        return x1, x2

    x1, x2 = sliders_and_data(cols[0], cols[1])

    # right, we show histograms
    bins = np.linspace(min([min(x1), min(x2)]),
                        max([max(x1), max(x2)]),
                        30)
    counts1, _ = np.histogram(x1, bins)
    counts2, _ = np.histogram(x2, bins)

    fig = plt.figure()
    plt.stairs(counts1, bins, color="blue", alpha=.85, fill=True, label='male')
    plt.stairs(counts2, bins, color="r", alpha=.65, fill=True, label='female')
    plt.grid()
    plt.close()
    cols[2].pyplot(fig)

    def KullbackLeibler_Discrete_binning(P, Q, threshold=.05, st=st):
        
        P, Q = P/sum(P), Q/sum(Q)  # normalize
        
        #method 1
        D = 0
        for p, q in zip(P, Q):
            if p > threshold:
                D -= p * np.log2(q/p)
        if D==np.inf:#np.isnan(D):
            st.caption('Method one yielded `inf` trying method 2...')
            # method 2
            D = 0
            for p, q in zip(P, Q):
                if q > threshold:
                    val = p * np.log2(p/q)
                    D += val
        return D
    D = KullbackLeibler_Discrete_binning(counts1, counts2, st=cols[2])
    cols[2].caption(f'Kullback Leibler divergence = {round(D,3)}')

    '---'
    '##### Model paramterization'
    text = r"""
    In choosing good model paramters, there are a couple things to consider;
    we should make use of physical insight, i.e., the parameters should actually be significantly descriptive of the data.

        $$
            \rho_{d,m} \gg 0
        $$
        
        Incorporate prior information about the problem to guide the choice of model parameterization.
        
        **number of parameters**: we should take care, as not to overdetermine or underdetermine the data. Utilize sparsity and parsimony to improve computational efficiency and generalizability.

    It's worth noting that choosing a good model parameterization is often a trade-off between the accuracy and the computational complexity of the solution. It's essential to find a balance between the two. 

    Upon parameterization, empirically validate the performance of the model parameterization by testing it on experimental or observational data.


    """
    
    wrapfig(width=320,src='https://caltech-prod.s3.amazonaws.com/main/images/TSchneider-GClimateModel-grid-LES-NEWS-WEB.width-450.jpg', text=text)

    '''
    *Most inverse problems that arise in practice are neither completely overdetermined nor completely underdetermined. For instance, in the X-ray tomography problem, there may be one box through which several rays pass (Fig. 3.10A). The X-ray opacity of this box is clearly overdetermined. On the other hand, there may be boxes that have been missed entirely (Fig. 3.10B). These boxes are completely undetermined. There may also be boxes that cannot be individually resolved because every ray that passes through one also passes through an equal distance of the other (Fig. 3.10C). These boxes are also underdetermined, since only their mean opacity is determined.* [sciencedirect.com]
    '''
    
def Probabilistic():
    ''
    '# Probabilistic inference'

    '##### Inverse problems in a Bayesian setting'
    text = r"""
        In Bayesian inference, an inverse problem is solved by inferring parameters from observed data and a prior probability distribution, which encodes prior information about the parameters. 

        The likelihood, $P(E|H)$, quantifies the probability of the data given the parameters.

        Together, the prior, $P(H)$, and likelihood functions define the posterior, $P(H|E)$ distribution over the parameters, which is the target of the inference. This distribution can be calculated using Bayes' theorem,

        $$ 
            P(H|E) = \frac{P(E|H)P(H)}{P(E)} \Rightarrow \sigma_\text{post} = \mathcal{L}\rho_\text{prior}
        $$
    """

    wrapfig(300, text, src='https://gowrishankar.info/blog/gaussian-process-and-related-ideas-to-kick-start-bayesian-inference/gp.png')


    r"""
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

    The Tarantola-Valette formalism is a specific implementation of the Bayesian formulation for inverse problems. Both approaches use a probabilistic formulation to estimate the unknown parameters of a system, and they both rely on Bayes' theorem to calculate the posterior distribution. However, the Tarantola-Valette formalism does not assume that the noise in the data is Gaussian nor that the prior and likelihood are Gaussian distributions.
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
    eps_space = np.logspace(-13, -9, 20)
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
    st.caption('Precision becomes high as resolution is increased.')
    
def Least_squares():
    def old():
        text_dict = getText_prep(filename = text_path+'Tikonov.md', split_level = 1)
        st.markdown(r'{}'.format(text_dict['1']))
    
    ''
    '# Least Squares'
    r"""
    ### Consider the determinedness of an inverse problem.
    

    ### What is the idea and the assumptions behind the least-squares solution?  
    The least-squares solution is a method for solving inverse problems that is based on the idea of minimizing the difference between the observed data and the model predictions. It is a deterministic method that finds a single, unique solution to the problem based on a set of predetermined equations.

    The basic idea behind the least-squares solution is to minimize the sum of the squares of the residuals, which are the differences between the observed data and the model predictions. Mathematically, this is represented by the following cost function:

    $$
        \mathcal{L}(m) = \frac{1}{2}(d-g(m))^T C_d^{-1}(d-g(m))
    $$

    Where $d$ is the observed data, $m$ is the vector of parameters, and $g(m)$ is the model.

    The least-squares solution is obtained by finding the values of m that minimize the cost function $\mathcal{L}(m)$. This can be done using optimization algorithms, such as gradient descent or Newton's method. The solution is the point estimate of the parameters that minimizes the sum of the squares of the residuals.

    The least-squares solution assumes that the noise in the data is Gaussian (but not neccessarily independent), and that the model is linear in the parameters (else the exponential of the cost function is no a Gaussian). Additionally, it assumes that the model is a good representation of the underlying physical process, and that the parameters are uniquely determined by the data. These assumptions are crucial for the method to work, and they may not always be valid in practice.

    The method is computationally efficient and easy to implement, and it provides a single point estimate of the parameters. However, it does not take into account the uncertainty in the data, the model or the parameters.

    ### Explain the Tikhonov solution to a linear inverse problem 

    The Tikhonov solution is a regularization method for solving linear inverse problems. It is a generalization of the least-squares solution that takes into account the uncertainty in the data and the model, and it is used to stabilize the solution and to prevent overfitting. The Tikhonov solution is based on the idea of adding a regularization term to the cost function, which penalizes large values of the parameters.

    The Tikhonov solution is obtained by minimizing the following cost function:

    $$ \mathcal{L}(m) = \frac{1}{2}(d-g(m))^T(d-g(m)) + \frac{\varepsilon}{2}m^Tm $$

    Where $d$ is the observed data, $g(m)$ is the model, $m$ is the vector of parameters, and $\varepsilon$ is a regularization parameter that controls the trade-off between the fit to the data and the smoothness of the solution. The first term in the cost function is the least-squares term, and the second term is the regularization term.

    The Tikhonov solution is obtained by finding the values of m that minimize the cost function $\mathcal{L}(m)$. This can be done using optimization algorithms such as gradient descent or Newton's method. The solution is a point estimate of the parameters that balances the fit to the data and the smoothness of the solution, depending on the value of the regularization parameter λ.

    The Tikhonov solution is particularly useful for linear inverse problems where the solution is ill-posed, meaning that the solution is not unique or stable. It is also used when the data is noisy or incomplete and the model is uncertain. The regularization term in the Tikhonov solution helps to stabilize the solution by adding prior information about the smoothness of the solution, which can be expressed by different norms such as L1 or L2. By controlling the value of the regularization parameter λ, it is possible to find a balance between fitting the data and avoiding overfitting.

    It is worth noting that the Tikhonov solution is also known as Ridge Regularization. It is a powerful tool for solving ill-posed linear inverse problems, but it requires the choice of the appropriate regularization term and the estimation of the regularization parameter. Additionally, it may not be suitable for non-linear or non-smooth problems.


    ### Is there a connection between a least-squares solution and a probabilistic solution? 

    Yes, there is a connection between a least-squares solution and a probabilistic solution. The least-squares solution is a deterministic method for solving inverse problems, which finds a single, unique solution by minimizing the sum of the squares of the residuals. On the other hand, a probabilistic solution is a statistical method that estimates the probability distribution of the parameters given the data and the model.

    However, the least-squares solution can be seen as a special case of the probabilistic solution, when the noise in the data is assumed to be Gaussian and independent, and the model is linear in the parameters. Under these assumptions, the least-squares solution is equivalent to the maximum likelihood estimate of the parameters.

    Additionally, the Tikhonov solution can be also seen as a special case of a probabilistic solution, when the regularization term is added to the likelihood function, this is known as a MAP (Maximum A posteriori) estimate, which is the mode of the posterior distribution.

    In summary, the least-squares solution and the Tikhonov solution are deterministic methods that can be seen as special cases of the probabilistic solution when certain assumptions are made about the noise and the model. Probabilistic methods provide a more general and flexible framework for solving inverse problems and take into account the uncertainty in the data and the model.

    
    ### Explain the concept of resolution for a linear problem
    
    
    The resolution of a linear inverse problem refers to the ability of the method to recover small and distinct features in the solution. In other words, it is a measure of how well the method can distinguish between different values of the parameters.

    The resolution of a method can be affected by various factors such as the noise level, the accuracy of the model, the dimensionality of the problem, and the regularization or stability of the method.

    In general, linear inverse problems tend to have a trade-off between resolution and stability, which means that increasing the resolution of the solution can lead to a less stable and more uncertain solution. There are several ways to increase the resolution of a linear inverse problem, such as by increasing the number of measurements, by improving the accuracy of the model, or by using regularization techniques.

    For example, the Tikhonov solution is a regularization method that can be used to increase the resolution of a linear inverse problem by balancing the fit to the data and the smoothness of the solution. By controlling the value of the regularization parameter, it is possible to increase the resolution of the solution without sacrificing stability.
    """

def Weakly_nonlinear():
    ''
    '# Weakly nonlinear problems and optimization'
    cols = st.columns(2)
    text_intro = """       
    A nonlinearity is characterized by: *a change of the output not being proportional to the change of the input*.
    
    Non-linear functions may appear chaotic, with troths and valleys scattered seemingly randomly. This makes non-linear phase-spaces tricky to optimize.


    Weakly non-linear problem typically only contain 1 or a few non-linearities, and crucially they may be approximated by a linear model in a neighborhood of a given point. 
    """

    cols[0].markdown(text_intro)
    fig = plt.figure()

    nsteps = 30#cols[0].slider('n', 1, 1000, 100)
    nparams = 6
    low, high = -4,4
    x = np.linspace(low,high,nsteps) 
    X, Y = np.meshgrid(x,x)

    params = [1, 1, 0.77, 1.11, .31, .31] # [cols[0].slider(l, 0.,10.,1.) for l in string.ascii_lowercase[:nparams]]


    
    def f(x,y):

        return abs((np.log(abs((x-a))**p) + np.log(abs((y-b))**p)))**(1/p)

    def f(x,y):
        return (x-a)**2 + (y-b)**2 + np.exp(d*(x-c)) + e*np.sin(y) 
    def pymathfunc2latex(string="np.sin(a*x)+(x-c)**2 + np.sin(y*a)-x**4 - y**2"):
        string = '$'+string+'$'
        replacements = {
            'np.s' : r"\s",
            '**': r'^ ' ,
            '*': r'\cdot ' 
        }
        for key, val in replacements.items():

            string = string.replace(key,val)

        return string

    #cols[1].write(pymathfunc2latex())

    def f(x,y, a=0, b=0, c=0, d=0, e=0, c1=0):
        ''
        val = a*np.sin(x*b) + c*np.cos(y*d)+e*x+c1*y +1/100*x**4+1/100*(y-1)**4+y/4
        
        #cols[0].write(pymathfunc2latex('a*np.sin(x*b) + c*np.cos(y*d)+e*np.tan(x*y*c1)'))
        return val+2


    z = f(X,Y, *params)

    


    def plot3dsurface():
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, z, color='white' ,
                           linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
            for elt in axis.get_ticklines() + axis.get_ticklabels():
                elt.set_visible(False)
            axis.pane.set_visible(False)
            axis.gridlines.set_visible(False)
            #axis.line.set_visible(False, )

        ax.set_facecolor('black')
        plt.close()
        fig.set_facecolor('black')
        #cols[1].pyplot(fig)

    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
                    x=X.flatten(), 
                    y=Y.flatten(),
                    z=z.flatten(),
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=z.flatten()-1,                # set color to an array/list of desired values
                        #colorscale='RdBu',   # choose a colorscale
                        opacity=0.8
                    )
                )])

    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=400, height=300,
                  margin=dict(l=0, r=0, b=0, t=0))
    cols[1].plotly_chart(fig, theme="streamlit", use_conatiner_width=True)
    

    '---'
    cols = st.columns(2)
    cols[0].markdown(
    """

    ##### Steepest decent for weakly non-linear optimization
    Steepest descent algorithms evaluate local gradient and step opposite with a step-length typically determined by a line search (for more info see scientific computing).


    """)
    
    x0 = np.random.uniform(-3,3,2)

    xs = [x0]
    h = 0.001
    lr = cols[1].select_slider('lr',np.round(np.logspace(-3,0,7), 3))

    nsteps = cols[1].slider('nsteps',1,100,10)
    for i in range(nsteps):
        x = xs[-1].copy()
        # determine grad
        
        y_current = f(x[0], x[1], *params)
        #y_current
        
        grad = np.array([ ( f(x[0]+h, x[1]+0, *params) - y_current ) / h,
                          ( f(x[0]+0, x[1]+h, *params) - y_current ) / h])

        #grad

        # step oposite
        x -= grad * lr
        xs.append(x)
    xs = np.array(xs)

    fig = plt.figure()
    
    plt.contourf(X, Y, z, 30, cmap='inferno')
    plt.plot(xs[:,0], xs[:,1])
    plt.scatter(x0[0], x0[1])
    plt.close()
    cols[1].pyplot(fig)
    cols[0].markdown("""
    
    
    ##### Convergence difficulties 

    Steepest descent algorithms have several convergence difficulties, such as: 
    * finding local minima instead of global minima, 
    * sensitivity to initial parameters,
    * non-convexity in cost function,
    * difficulty in choosing the right step size and getting stuck in a plateau of cost function.
    
    More advanced optimization techniques such as conjugate gradient or Newton-Raphson method, mitigate some of these challenges. 

    """)
    '---'
    """
    ##### Challenges to the efficiency (speed) of steepest decent algorithms when solving weakly non-linear problems? 

    There are several challenges to the efficiency (speed) of steepest descent algorithms when solving weakly non-linear problems:

    * High-dimensional problems: The steepest descent algorithm can be slow to converge for high-dimensional problems, as the cost function may have many local minima and the algorithm needs to explore the parameter space extensively to find the global minimum.
    * Computational complexity: The steepest descent algorithm requires the calculation of the gradient of the cost function at each iteration, which can be computationally expensive for large or complex models.
    * Line search: The steepest descent algorithm requires a line search to determine the step size at each iteration. This process can be computationally intensive and slow the algorithm down.
    """

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
    '# Monte Carlo'
    r"""
    ### For which type of inverse problems are Monte Carlo (MC) methods used? 
    Monte Carlo (MC) methods are often used for inverse problems that involve complex or high-dimensional systems, where the likelihood function and/or the prior distribution cannot be easily calculated or integrated analytically. These methods involve generating a large number of random samples from the prior distribution, and then evaluating the likelihood function for each sample. The samples that have a high likelihood are considered to be more probable, and they are used to estimate the posterior distribution. A typical likelihood is the following:

    $$
        L(m) = k\exp\left[1/2 (g(m) - d_\text{obs})^T C^{-1} (g(m) - d_\text{obs})^T\right]
    $$

    Markov Chain Monte Carlo (MCMC) methods are a class of MC methods that are particularly useful for inverse problems. These methods involve generating a sequence of samples that are correlated with each other, and they are designed to converge to the true posterior distribution.

    Additionally, these methods can also be used to calculate various quantities of interest, such as credible intervals or the expected value of a function of the parameters.


    **Neighbourhood** in monte carlo, refers to a single iteration step: it is all the points in the parameters space we can reach from the current parameter configuration with the given step size.


    Accept reject, describes which steps we choose to keep. A typical acceptance criterion is the Metropolis-hastings algorithm, but not assuming uniform prior distribution:
    
    $$
    \begin{align*}
        P_\text{accept} = 1 && or && \exp(\Delta S/\sigma^2 )
    \end{align*}
    $$

    ### What is the difference between an MC solution and a "deterministic" solution? 
    An "deterministic" solution of an inverse problem refers to a method that finds a single, unique solution to the problem based on a set of predetermined equations or algorithms. These methods typically involve optimization algorithms or analytical solutions that are designed to find the best fit of the model to the data. They provide a point estimate of the parameters of the system, which is a single set of parameter values that maximizes the likelihood or minimizes some cost function.

    On the other hand, a Monte Carlo (MC) solution of an inverse problem refers to a method that generates a large number of possible solutions, called samples, and the solution is represented by a probability distribution over the parameter space, rather than a single point estimate. These methods involve generating a large number of random samples from the prior distribution, and then evaluating the likelihood function for each sample. The samples that have a high likelihood are considered to be more probable, and they are used to estimate the posterior distribution.

    ### What kind of information about the solution of an inverse problem can be obtained by an MC method? 

    A Monte Carlo (MC) method for solving an inverse problem generates a large number of possible solutions, called samples, and the solution is represented by a probability distribution over the parameter space, rather than a single point estimate. By analyzing the samples, it is possible to obtain various types of information about the solution of the inverse problem, such as:

    * Point estimates: The most likely values of the parameters can be estimated by finding the peak of the posterior distribution, or by calculating the mean or median of the samples.

    * Uncertainty: The uncertainty of the parameters can be quantified by analyzing the spread of the samples, such as by calculating the standard deviation, interquartile range, or credible intervals.

    * Correlations: The correlation between different parameters can be analyzed by calculating the covariance or correlation matrix of the samples.

    * Model validation: The model can be validated by comparing the predicted data with the observed data, and by analyzing the residuals or the prediction intervals.

    * Model selection: The relative evidence for different models can be analyzed by comparing the posterior probabilities or the Bayesian Information Criterion (BIC) of the models.

    * Sensitivity analysis: The sensitivity of the solution to different inputs can be analyzed by perturbing the data or the model and by analyzing the effect on the samples.

    * Visualization: The samples can be visualized in different ways, such as by plotting the histograms, scatter plots, or density plots, which can help to understand the shape of the posterior distribution and the patterns in the data.

    In summary, Monte Carlo methods can provide a lot of information about the solution of an inverse problem, including point estimates, uncertainty, correlations, model validation, model selection, sensitivity analysis, and visualization.


    ### What are the weaknesses of MC methods when only a finite number of samples can be produced? 

    When only a finite number of samples can be produced, Monte Carlo methods can have weaknesses such as:

    * Convergence issues and inaccuracies in approximating the posterior,
    * Poor mixing and slow convergence of the Markov Chain,
    * High auto-correlation between samples leading to underestimation of uncertainty,
    * Computational challenges for high-dimensional systems,
    * Difficulty in distinguishing between different models,
    * Sensitivity to the choice of the prior distribution.

    In summary, MC methods can provide a lot of information about the solution of an inverse problem, but their accuracy and reliability may be limited when only a finite number of samples can be produced. It is important to carefully design the sampling strategy and to analyze the samples in order to understand the convergence and the uncertainty of the solution.
    """
    sphereINcube_demo()

# Navigator
topic_dict = {
    "Landing Page" : landingPage,
    'Information theory' : informationTheory,
    'Probabilistic inference' : Probabilistic,
    'Monte Carlo': monteCarlo,
    'Least-squares / Tikonov': Least_squares, 
    'Weakly nonlinear problems and optimization' : Weakly_nonlinear, 
    'Density variations (Tikonov)': DensityVar_LeastSquare,
    'Linear Tomography (Tikonov)' : ass1,
    'Vertical Fault (Monte Carlo)': ass2,
    'Glacier thickness (Monte Carlo)' : ass3_glacier_thickness,
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()


p.stop()
p