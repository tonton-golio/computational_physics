from utils_complex import *
devmod = False

# Pages
def homeComplex():
    text = getText_prep(textfile_path+'home.md',3)
    st.title('Complex Physics')
    col1, col2 = st.columns(2)
    col1.markdown(text['col1'])
    col2.markdown(text['col2'])

def statisticalMechanics():
    # Detailed Description
    ## some of these should perhaps be partially unpacked
    st.markdown(r"""# Statistical Mechanics""")
    text_dict = getText_prep(filename = textfile_path+'statisticalMechanics.md', split_level = 2)
    #a = text_dict.keys()
    #a

    text_expander(key="Microcanonical Ensemble", 
            text_dict=text_dict, expanded=False)
    text_expander(key="Canonical Ensemble", text_dict=text_dict)

    key="Metropolis algorithm"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key = "Ising Model "
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])
    

    # Simulation
   
    st.markdown(r"""
        ### Ising 2d Simulation  
        Snapshots of the output of a simulation of the 2d Ising model using the metropolis algorithm.
        """)
    cols = st.columns(3)
    size = cols[0].slider('size',3,100,10)
    beta = cols[1].slider('beta',0.01,5.,1.)
    nsteps = cols[2].slider('nsteps',3,10000,100)
    

    nsnapshots = 4 #  multiples of 4 
    results, data = ising(size, nsteps, beta, nsnapshots)
     
    
    st.pyplot(plotSnapshots(results, nsnapshots))

    st.markdown(r"""A time series indicates whether we have entered steady-state, and the susceptibility plots indicates (🤞) the phase-transition. *Phase transitions are trypically soft in the small $L$ regime.*""")
    cols = st.columns(2)
    cols[0].pyplot(plotEnergy_magnetization(results))
    cols[1].pyplot(plotSusceptibility(data))
     
def phaseTransitions_CriticalPhenomena():
    text_dict = getText_prep(filename = textfile_path+'phaseTransitions.md', split_level = 2)
    #a = text_dict.keys(); a

    st.title('Phase Transitions and Critical Phenomena')
    
    key = 'Mean-field Hamiltonian'
    st.markdown(text_dict[key])

    key = 'Mean-field Hamiltonian (derivation)'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
    
    key = 'MF: Z, m, Tc, F & critical exponent'
    st.markdown(text_dict[key])
    
    key = 'MF: Z, m, Tc, F & critical exponent (derivation)'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    
    st.markdown(r"""
    Let's have a look at how this approximation compares to the typical approach. I'll run the 1d Ising with NN with and without the mean-field approximation.
    """)
    #with st.sidebar:
    cols = st.columns(3)
    size = cols[0].slider('size',3,100,30)
    beta = cols[1].slider('beta',0.01,5.,1.5)
    nsteps = cols[2].slider('nsteps',3,10000,100)

    
    fig, _ = ising_1d(size, beta, nsteps)
    st.pyplot(fig)

    st.markdown(r"Hmmm, my implementation is probably bad...")

    key = '1D Ising model and transfer matrix method'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
      
def percolation_and_fractals():

    text_dict = getText_prep(filename = textfile_path+'percolation_and_fractals.md', split_level = 3)
    #a = text_dict.keys(); a

    # Side bar
    with st.sidebar:
        st.markdown('**Extra Paramteres**') 
        with st.expander('square grid percolation'):

            cols_sidebar = st.columns(2)
            size = cols_sidebar[0].slider('size', 4  , 64, 24)
            n_ps = cols_sidebar[1].slider('Num ps = ', 5,20,5)
            marker_dict = {'point': '.','pixel': ',',}
            marker = marker_dict[cols_sidebar[0].radio('marker', marker_dict.keys())]
            seed = cols_sidebar[1].slider('seed',10,100)

    # Render
    st.markdown(r"""# Percolation and Fractals""")
    
    st.markdown(r"""## Percolation""")
    cols = st.columns(2)
    cols[0].markdown(r"""
    Percolation theory considers the behavior of a network. An appropriate case to start with is a 2D lattice with nearest neigbour connections. Let the nodes (intersections) require activation energy $p$ to be *open*. Neighboring open sites connect to form domains.

    On the right, is a randomly generated grid. Vary $p$ to alter openness and affect the number of domains, $N(p)$.
    """)
    
    p_percolation = cols[0].slider("""p =""",       0.01, 1. , .1)
    fig_percolation, domains = percolation(size, seed, p_percolation,marker, devmod=False)
    #domains
    cols[1].pyplot(fig_percolation)
    cols[0].latex(r"""N({}) = {}""".format(p_percolation, len(domains)))
    
    fig = percolation_many_ps(n_ps, size, seed)
    cols[1].pyplot(fig)


    st.markdown(r"""By visualizing $N(p_c)$, we find that this lattice structure undergoes a phase transition at $p_c=\frac{1}{2}$. An easy way to notice this transition, and see just how *hard* it is, is to look at domain weight distribution:
    """)
    with st.expander('video of percolation', expanded=False):
        st.video('assets/complex/images/percolation_animation.mp4')
    st.markdown(r"""Notice, at $p=p_c=\frac{1}{2}$ a domain starts to dominate.""")


    # Bethe lattice
    st.markdown(r"""
    ## Bethe Lattice""")
    cols = st.columns(2)
    cols[0].markdown(r"""
    Bethe lattice (regular tree)  is an infinite connected 
    cycle-free graph where all vertices have the same number of neighbors.  
    """)
    degree = cols[0].slider("""degree""",       2, 5 , 3)
    
    st.markdown(r"""
    We may perform percolation on this lattice. To do this, we fill the adjencancy matrix, not with boolean value, but instead with random samples drawn from a uniform distribution.
    """)
    cols[1].pyplot(betheLattice(0, size={2:5,3:10,4:17,5:26}[degree], degree=degree))

    #st.graphviz_chart(betheLattice_old())
    size_beth = 40
    #cols=st.columns(2)
    p_beth = st.slider("""p = """,       0.01, 1. , .33)
    
    st.pyplot( betheLattice(p_beth, size=size_beth, degree=degree))
    
    st.markdown(r'''Again we may take a look at the number of domains as a function of $p$.''')

    
    Ns = betheLattice(size=32, get_many=True, 
                    ps=np.linspace(.1,.9,7), degree=degree)
        
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(Ns.keys(),Ns.values() , c='white')
    ax.set_xlabel(r'$p$', color='white')
    ax.set_ylabel(r'Number of domains, $N$ ', color='white')
    st.pyplot(fig)

    st.markdown(r"""We may find the critical point for different degree, $z$, bethe lattices.
    $$
        p_c = \frac{1}{z-1}
    $$
    """)

    st.markdown(r"""## Mandelbrot""")
    st.markdown(r"""
    The Mandelbrot set contains complex numbers remaining stable through the mandelbrot function after successive iterations. Note; we let $z_0$ be 0. The two main essential pieces of code are displayed below the plot.
    """)
    cols = st.columns(3)
    size_fractal = int(10**cols[0].slider(r'Resolution (log)',1.5,4., 2.))
    
    n = cols[1].slider('n',1,50,27)
    a = cols[2].slider('a',0.01,13.,2.3)
    fig, res = run_fractals(size_fractal, a, n)
    st.pyplot(fig)
    st.markdown(r"""
    To optimize I assume the output is symmetric across the real axis. 
    $$
    \textbf{stable}( \textbf{mandelbrot}(a-ib) ) = \textbf{stable}( \textbf{mandelbrot}(a+ib))
    
    $$
    """)
    
    def code(lvl, text):
        lines = text.split('\n')[1:]
        for i, line in enumerate(lines):
            lines[i] = line[4*lvl:]
        return '\n'.join(lines)

    cols = st.columns(2)
    cols[0].code(code(2,r"""
        def stable(z):
            try:
                return False if abs(z) > 2 else True
            except OverflowError:
                return False
        """))
    
    cols[1].code(code(2,r"""
        def mandelbrot(c, a=2, n=50):
            z = 0
            for i in range(n):
                z = z**a + c
            return z
        """))
    st.markdown(r"""
    ## Fractal Dimension
    After we get the formulae for this, we could look at the fractal dimension of the mandelbrot set at different zoom-levels, and find out whether its scale free.


    """)
    fig, fig2 = fractal_dimension(res)
    st.pyplot(fig)
    st.pyplot(fig2)
    st.markdown('### Resources\n' + text_dict['Resources'])

def selfOrganizedCriticality():
    st.title("Self organized criticality (SOC)")
    
    text_dict = getText_prep(filename = textfile_path+'selfOrganizedCriticality.md', split_level = 2)
    if devmod: a = text_dict.keys(); a

    st.markdown(text_dict['Intro'])
    
    key = 'Bak-Sneppen'
    st.markdown('#### ' + key)
    st.markdown(text_dict[key])
    # Main fig
    cols = st.columns(3)
    nsteps = cols[0].slider('nsteps',1,30000,5000)
    size = cols[1].slider('size',10,1000,300)
    random_func = cols[2].selectbox('random func',['uniform', 'normal'])
    
    chains, idx_arr, L = bakSneppen(size, nsteps, random_func)
    st.pyplot(bakSneppen_plot_imshow(chains, size, nsteps))

    # fill
    cols = st.columns(2)
    key = 'Bak-Sneppen 2'
    cols[0].markdown(text_dict[key])

    skip_init = skipInit(chains, patience=1000, tol=0.0001)
    avalanches_dict = avalanches(idx_arr, skip_init)

    fig_baksneppen_fill = bakSneppen_plot_initial(chains, skip_init, idx_arr)
    cols[1].pyplot(fig_baksneppen_fill)
    cols = st.columns(2)
    cutoff1 = cols[0].slider('cutoff1', -5,5,0)
    cutoff2 = cols[1].slider('cutoff2', -5,5,0)
    st.pyplot(plotAvalanches(idx_arr, skip_init, avalanches_dict, cutoff1, cutoff2))
    

    st.markdown(r"""
    ## Branching
    Branching is a stochastic process which let allow for continue evolution, or death. Random walks are an example of branching.

    In 1 dimension, we either take a step right or a step left at each iteration. It should be very likely that we return to the origin, however some *walks* may take us on a long trip.
    """)
    run_firstReturn1D = st.button('run: First return of 1d randomwalk')
    if run_firstReturn1D: firstReturn1D()

    st.markdown(r"""
    In 2 dimensions, we generate a random angle and take a step in that direction.
    """)
    st.code(r"""
    def randomWalk(nsteps):
        for i in range(nsteps):
            theta = random()
            dx = np.cos(theta) ; x += dx
            dy = np.sin(theta) ; y += dy 
        """)
    st.markdown(r"""
    When looking for first return in 2d with floating point angles, we must consider a region around origo to be *home*. I implement this by looking when the agent enters the unit circle after having been a distance greater than $\frac{3}{2}$ from the origin.
    """)
    run_firstReturn2D = st.button('run: First return of 2d randomwalk')
    if run_firstReturn2D: firstReturn2D()
    
    
    st.markdown(r"""
    Notice, it becomes increasingly unlikely to make a swift return as we add dimensions to our random walk model.""")

    st.markdown(r"""## Sandpile Model (bereaucrats)""")

    

    st.markdown(r"""
    The problem with beraucrats, is that they dont finish tasks. When a task 
    lands on the desk of one, the set a side to start a pile. When that pile contains 
    4 tasks, its time to distribute them amongst the nieghbors. If a
    beraucrat neighbours an edge, they may dispose of the task headed in that direction. 
    """)

    cols = st.columns(3)
    run_bereaucrats = cols[0].button('run: bereaucrat model')
    size_bereaucrats = cols[1].slider('size_bereaucrats', 10,100,30)
    nsteps_bereaucrats = cols[2].slider('nsteps_bereaucrats', 10,20000,3000)
    
    if run_bereaucrats: bereaucrats(nsteps_bereaucrats, size_bereaucrats)

    st.markdown(r"""
    After a while, the model reaches steady state. Analyzing the distribution of avalanche sizes in steady state, reveals the ...
    """)
    st.image('assets/complex/images/sandpile_2d_aval_dist.png')
    
    #st.markdown('## Evolution Model')

def networks():
    st.title('Networks')
    text_dict = getText_prep(textfile_path+'Networks.md', 3)

    st.markdown(text_dict['Intro'])
    cols = st.columns(2)
    cols[0].markdown(r"**Social-networks**")
    cols[0].markdown(text_dict['Social-networks'])
    cols[1].markdown(r"**Paper-authors**")
    cols[1].markdown(text_dict["Paper-authors"])

    with st.expander(r"Metrics and features of networks"):
        st.markdown(text_dict["Metrics and features of networks"])

    cols = st.columns(2)
    network_type = cols[0].radio('network type',['bethe', 'random'])
    N = cols[0].slider('N',1,420,22)
    if network_type == 'random':
        a = cols[0].slider('alpha', 0.9,1.,0.99)
    if network_type == 'bethe':
        degree = cols[0].slider('degree', 2,5,3)
        
    net = make_network(N,a) if network_type == 'random' else makeBetheLattice(N,degree)
    fig, G = draw_from_matrix(net)
    cols[1].pyplot(fig)


    # lets take a look at the degree dist

    fig = network_analysis(net, G)
    st.pyplot(fig)

def agent_event_models():
    st.title('Agent/event based models')
    text_dict = getText_prep(filename = textfile_path+'agentbased.md', split_level = 3)
    

    key = 'Game of life'
    cols = st.columns(2)
    cols[0].markdown(text_dict[key])
    initial_config = cols[1].selectbox('initial config', ['glider','square', "boat", "loaf", "ship"])
    st.pyplot(game_of_life(initial_config=initial_config))
    st.markdown(text_dict['Game of life 2'])

    key = 'Stochastic simulation'
    st.markdown('### '+key)
    st.markdown(text_dict[key])
    
    class Agent():
        def __init__(self, pos, type='Rabbit', mapsize=10):
            self.pos = pos
            self.type = type
            self.mapsize = mapsize

        def step(self):
            (update_x, update_y) = (np.random.choice([-1,0,1]) for _ in range(2))
            self.pos[0] = self.pos[0] + update_x if ((self.pos[0] + update_x) >=0) and ((self.pos[0] + update_x) <self.mapsize) else self.pos[0]

            self.pos[1] = self.pos[1] + update_y if ((self.pos[1] + update_y) >=0) and ((self.pos[1] + update_y) <self.mapsize) else self.pos[1]

        def __repr__(self) -> str:
            return f"{self.type} at {tuple(self.pos)}"

    # Init
    size = 10
    
    agents = []
    foxes = [(0,1), (2,2)]
    for f in foxes: 
        agents.append(Agent(pos=list(f), type='Fox'))
    rabbits = [(4,1), (1,5), (3,9), (8,7), (9, 1), (5,5)]
    for r in rabbits: 
        agents.append(Agent(pos=list(r), type='Rabbit'))
    
    #agents
    # fill map
    X = np.zeros((size,size))
    for agent in agents:
        
        X[tuple(agent.pos)] = 1 if agent.type=='Rabbit' else 2

    fig, ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].imshow(X, aspect=.4)

    # step
    for agent in agents:
        agent.step()
        
    # fill map
    occupied_positons = []
    X = np.zeros((size,size))
    for agent in agents:
        
        X[tuple(agent.pos)] = 1 if agent.type=='Rabbit' else 2
        occupied_positons.append(tuple(agent.pos))

    ax[1].imshow(X, aspect=.4)

    

    #plt.grid()
    
    st.pyplot(fig)

    """ **Rules**
    * at each timestep, all agents step to a random nieghboring site.
    * if a rabbit and fox meet, the fox will eat the rabbit
    * if two rabbits meet, they will procreate
    * if a fox hasn't eaten in 3 days, it will die
    * if foxes meet, they will procreate
    """


    """
    Hmm, question is: *should I do this object oriented style, or with matricies?*

    I'll come back to this. Main take away is that this is a different and usually more realistic method for time-evolving dynamic systems than just applying forward-Euler to the partial differential equations.
    """

    others = ['Gillespie algorithm, ', 'Example of agent based simulation', 'Advantages of agent based models']
    for key in others:
        st.markdown('### '+key)
        st.markdown(text_dict[key])
    
def econophysics():
    st.title('Econophysics')
    text_dict = getText_prep(filename = textfile_path+'econophysics.md', split_level = 3)
    
    st.markdown('#### Brownian Motion\n' + text_dict['Brownian Motion'])
    cols=st.columns(2)
    cols[0].markdown(r'If we pull a stock')
    ticker = cols[1].selectbox('Ticker',['GOOGL', 'AAPL', 'TSLA'])
    fig, timeseries = var_of_stock(ticker = ticker)
    st.pyplot(fig)

    st.markdown(text_dict['Brownian Motion 2'])


    st.markdown('#### Hurst exponent\n' + text_dict['Hurst exponent'])
    fig = hurstExponent(timeseries)
    st.pyplot(fig)
    st.markdown('#### Fear-factor model\n' + text_dict['Fear-factor model'])
    st.markdown('#### Bet-Hedghing Model\n' + text_dict['Bet-Hedghing Model'])

    cols = st.columns(2)
    noise = cols[0].slider('noise',0.,3.,1.)
    p = cols[0].slider('p', 0.,1.,.05) 
    win_multiplier = cols[0].slider('win multiplier', 1.,4.,1.5) 
    loss_multiplier = cols[0].slider('loss multiplier', 0.,1.,.5) 

    invest_per_round = cols[0].slider('invest per round', 0.,1.,.5) 
    nsteps = cols[0].slider('nsteps    ',1,3000,500)
    
    fig = betHedging(p, noise, invest_per_round, nsteps, win_multiplier, loss_multiplier)
    cols[1].pyplot(fig)

# Navigator
topic_dict = {
    'Contents' :                               homeComplex,
    'Statistical Mechanics' :                  statisticalMechanics,
    'Phase transitions & Critical phenomena' : phaseTransitions_CriticalPhenomena,
    'Percolation and Fractals' :               percolation_and_fractals,
	'Self-organized Criticality' :             selfOrganizedCriticality,
    'Networks' :                               networks,
    'Agent based models' :               agent_event_models,
    'Econophysics'  :                          econophysics, }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()