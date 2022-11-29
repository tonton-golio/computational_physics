from utils.utils_appstat import *

from scipy.optimize import curve_fit

def home():
    st.title('Applied statistics')
    st.image('assets/images/Stable_diffusion__Mathematician_discovering_Neptune.png', width=420)
    st.caption('Stable diffusion response to prompt: *Mathematician discovering Neptune* [[stable diffusion]](https://huggingface.co/spaces/stabilityai/stable-diffusion).')

def week1():
    
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
     
    st.header(text_dict['title'])
    with st.expander('Week 1 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    with st.expander('Mean', expanded=False):
        st.markdown(text_dict['Mean'])
        
        mean_picker = st.selectbox('There are different measures hereof',['Geometric mean', 'Arithmetic mean', 'Median',  'Mode', 'Harmonic', 'Truncated'])

        cols = st.columns(2)
        cols[0].markdown(text_dict[mean_picker])
        cols[1].markdown(text_dict[mean_picker+' code'])
        
        '##### Demo'
        
        cols = st.columns(5)
        n_normal = cols[0].slider("n_normal", 1, 1000, 100)
        n_exp= cols[1].slider("n_exp", 1, 1000, 0)
        n_cauchy= cols[2].slider("n_cauchy", 1, 1000, 0)
        truncate = cols[3].slider("truncate", 1, 100, 1)
        seed = cols[4].slider("seed", 1, 100, 42)
        np.random.seed(seed)
        normal = np.random.randn(n_normal)
        exp = np.random.exponential(n_exp) 
        cauchy = np.random.standard_cauchy(n_cauchy)
        
        arr = np.hstack([normal, exp, cauchy]).flatten()

        fig = showMeans(arr, truncate)   # Perhaps change this to 2D
        st.pyplot(fig)


    with st.expander('Standard Deviation', expanded=False):
        st.markdown(text_dict['STD'])

        # compare N-1 and N...
        #if st.button('run std_calculations'):
        st.pyplot(std_calculations())


    st.markdown(text_dict['Weighted mean'])

    with st.expander('Correlation', expanded=False):
        st.markdown(text_dict['Correlation'])

    with st.expander('Central limit theorem', expanded=False):
        
        cols = st.columns(2)
        cols[0].markdown(text_dict['Central limit theorem'])
        
        # roll dice
        #cols[1].pyplot(roll_dice())

        # roll a die
        cols[1].pyplot(roll_a_die(420))
        cols[1].caption('rolls of a die')

    
        st.markdown(text_dict['Central limit theorem 2'])
        
        
        cols = st.columns(4)

        n_points = [cols[i].slider('n_'+name,10,2000, 100) for i, name in enumerate('uniform, exponential, chauchy'.split(', '))]
        n_experiments = cols[3].slider("N_experiments",1, 1000, 100)

        sums = makeDistibutions(n_experi = n_experiments, 
                                N_uni= n_points[0], 
                                N_exp= n_points[1], 
                                N_cauchy = n_points[2])

        
        st.pyplot(plotdists(sums))

        
       

        # combination
        #st.markdown(f'''
        #    |num. $\sigma$   | inside %, $p$, theo.  | N_concat: {sum(n_points)} | N_gauss: {n_points[0]} | N_exp: {n_points[1]} | N_chaucy: {n_points[2]}  |
        #    |---|---|---|---|---|---|
        #    |1   |68         | {sigs_lst[3,0]} | {sigs_lst[0,0]}  |{sigs_lst[1,0]}  |{sigs_lst[2,0]}  |
        #    |2   |95         | {sigs_lst[3,1]} | {sigs_lst[0,1]}  | {sigs_lst[1,1]} | {sigs_lst[2,1]} |
        #    |3   |99.7       | {sigs_lst[3,2]} | {sigs_lst[0,2]}   | {sigs_lst[1,2]} |  {sigs_lst[2,2]}|
        #    |4   |99.99995   | {sigs_lst[3,3]} | {sigs_lst[0,3]}   |{sigs_lst[1,3]}  | {sigs_lst[2,3]} |
        #''')
        
        

        st.markdown(text_dict['Central limit theorem 3'])

    with st.expander('Error propagation', expanded=False):
        st.markdown(text_dict['Error propagation'])
        cols = st.columns(2)
        cols[0].markdown(text_dict['Error propagation add'])
        cols[1].markdown(text_dict['Error propagation mul'])

        st.markdown(text_dict['Demo'])
        demoArea()
        
        
        st.markdown(text_dict['Error propagation 2'])


      
    st.markdown(text_dict['Estimating uncertainties'])
    with st.expander('ChiSquare method, evaluation, and test', expanded=False):
        st.markdown(text_dict['ChiSquare method, evaluation, and test'])

        # fit binned data
        x = np.random.randn(100)*1.1+0.2
        Nbins = 20
        xmin, xmax = -3,3
        sy = 1 # for now, to avoid error, insert correct
        def func_gauss():
            pass
        
        def chi2_owncalc(N,mu,sigma):
            y_fit = func_gauss(x,N,mu,sigma)
            chi2 = np.sum(((y-y_fit) / sy)**2)
            return chi2

        # fit minuit

        fig, ax = plt.subplots()
        counts, bins = np.histogram(x, bins = Nbins, range=(xmin, xmax))

        ax.hist(x)
        plt.close()
        st.pyplot(fig)

        # note: fix, number of decials in nice text plot in External functions.

    with st.expander('New ChiSquare method', expanded=False):
        'make random data'
        x_range = np.linspace(0,10, 100)
        sample = np.random.normal(3,2,100)
        # Define signal PDF:
        def gauss_pdf(x, mu, sigma) :
            """Gaussian"""
            return 1.0 / np.sqrt(2*np.pi) / sigma * np.exp( -0.5 * (x-mu)**2 / sigma**2)
        





    st.markdown(text_dict['Links'])
    
def week2():
    # Import text dicts
    text_dict = getText_prep(filename = text_path+'week2.md', split_level = 1)
    text_dict3 = getText_prep(filename = text_path+'week2.md', split_level = 3)

    # Header and description
    st.header('PDFs, Likelihood, Systematic Errors')
    with st.expander('Week 2 description', expanded=False):
        st.markdown(text_dict['description'])

    # PDFs
    with st.expander('PDFs', expanded=False):
        st.markdown(text_dict['Header 1'])

    # distributions 
    with st.expander('Distributions', expanded=False):
        dist = st.selectbox('Picker:', "Binomial, Poisson, Gaussian, Student's t-distribution".split(', '))
        st.markdown(text_dict3[dist])

    # View distributions
    #with st.expander('View distributions', expanded=False):
        st.pyplot(PDFs(st.slider('number of experiments:', 1,999,69)))

    # Maximum likelihood estimation
    st.markdown(text_dict['maximum likelihood'])
    cols = st.columns(3)
    mu =          cols[0].slider('mu', -2.,2.,0.)
    sig =         cols[1].slider('sig', 0.,5.,1.)
    sample_size = cols[2].slider('sample_size', 1,10000,120)
    
    sample = np.random.normal(loc=mu,scale=sig, size=sample_size)

    mu, sig, L, fig = maximum_likelihood_finder(mu, sample, return_plot=True, verbose=True)
    st.pyplot(fig)

    st.markdown(text_dict['maximum likelihood 2'])

    
    N_random_sample_runs = st.slider('number of random sample runs', 10, 100, 23)
    

    fig, prob_worse = evalv_likelihood_fit(mu, sig, L, sample_size, N_random_sample_runs)
    st.pyplot(fig)
    st.write('probability of worse:', prob_worse)

    
def week3():
    text_dict = getText_prep(filename = text_path+'week3.md', split_level = 1)
     
    #st.header('Week 3')
    with st.expander('Week 3 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def week4():
    #st.header('Week 4')
    text_dict = getText_prep(filename = text_path+'week4.md', split_level = 1)
     
    with st.expander('Week 4 description', expanded=False):
        st.markdown(text_dict['description'])
    st.markdown(text_dict['Header 1'])

    def random(size, dist = 'normal', mu=0, sigma=1):
        return [i for i in range(size)]

def week5():
    #st.header('Week 5')
     
    text_dict = getText_prep(filename = text_path+'week5.md', split_level = 1)
    with st.expander('Week 5 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def week6():
    #st.header('Week 6')

    text_dict = getText_prep(filename = text_path+'week6.md', split_level = 1)
    with st.expander('Week 6 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    X = makeBlobs(100)
    st.pyplot(scatter(X[:,0], X[:,1]))
     
def week7():
    #st.header('Week 7')
    
    text_dict = getText_prep(filename = text_path+'week7.md', split_level = 1)
    
    with st.expander('Week 7 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def utils_explorer():
    st.markdown(r'''
    ## Utils explorer
    below are the *coore functions* and *extra functions*.
    ''')
    func_dict_core, func_dict_extras = makeFunc_dict(filename='utils/utils_appstat.py')

    func_name = st.selectbox('function (core)', func_dict_core.keys())
    st.code(func_dict_core[func_name] )
    func_name_extra = st.selectbox('function (extras)', func_dict_extras.keys())
    st.code(func_dict_extras[func_name_extra] )
    
# Navigator
topic_dict = {
    'Welcome': home,
    'Intro & ChiSquare': week1,
    'Likelihood & Sys. Errors': week2,
    #'week 3': week3,
    #'week 4': week4,
    #'week 5': week5,
    #'week 6': week6,
    #'week 7': week7,
    'Utils explorer' : utils_explorer,
    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()