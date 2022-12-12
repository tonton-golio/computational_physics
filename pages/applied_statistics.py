from utils.utils_appstat import *

from scipy.optimize import curve_fit
#st.markdown('<div align="center"> Hello</div>' , unsafe_allow_html=True)


fig_counter[0] = 0

def home():
    st.title('Applied statistics')
    st.image('assets/images/Stable_diffusion__Mathematician_discovering_Neptune.png', width=420)
    st.caption('*Mathematician discovering Neptune* [[AI-generated with stable diffusion]](https://huggingface.co/spaces/stabilityai/stable-diffusion).')
    st.markdown("""
        Course taught by: Troels C. Petersen. 

        [course website](https://www.nbi.dk/~petersen/Teaching/AppliedStatistics2022.html)""")
    from streamlit.components.v1 import html
    my_html = """<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1399648774&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/antartica_traffic_control" title="Goblin Mode" target="_blank" style="color: #cccccc; text-decoration: none;">Goblin Mode</a> · <a href="https://soundcloud.com/antartica_traffic_control/other-eyes" title="Other Eyes" target="_blank" style="color: #cccccc; text-decoration: none;">Other Eyes</a></div>"""

    html(my_html)
def week1():
    
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
     
    st.header(text_dict['title'])
    with st.expander('Week 1 description', expanded=False):
        st.markdown(text_dict['description'])

    with st.expander('Mean', expanded=False):
        st.markdown(text_dict['Mean'])
        
        mean_picker = st.selectbox('There are different measures hereof',['Geometric mean', 'Arithmetic mean', 'Median',  'Mode', 'Harmonic', 'Truncated'])

        cols = st.columns(2)
        cols[0].markdown(text_dict[mean_picker])
        cols[1].markdown(text_dict[mean_picker+' code'])
        
        '##### Demo'
        'Select number of points from each distribution, as well as, how many to truncate.'
        
        demo_comparing_means()

    with st.expander('Standard Deviation', expanded=False):
        cols = st.columns(2)
        cols[0].markdown(text_dict['STD'])
        n = 400
        cols[1].pyplot(std_calculations(n))
        caption_figure(f'Standard deviation with two different methods. N is the sample size, and I have done this {n} times.', cols[1])


    st.markdown(text_dict['Weighted mean'])

    with st.expander('Correlation', expanded=False):
        st.markdown(text_dict['Correlation'])

    with st.expander('Central limit theorem', expanded=False):
        
        
        st.markdown(text_dict['Central limit theorem intro'])
        
        # roll dice
        #cols[1].pyplot(roll_dice())

        # roll a die
        st.pyplot(roll_a_die(200))
        caption_figure('rolls of a die')

        st.markdown(text_dict['Central limit theorem'])
        st.markdown(text_dict['Central limit theorem 2'])
        
        
        cols = st.columns(4)
        n_points = [cols[i].slider('n_'+name,10,2000, 100) for i, name in enumerate('uniform, exponential, chauchy'.split(', '))]
        n_experiments = cols[3].slider("N_experiments",1, 1000, 100)

        sums = makeDistibutions(n_experiments, *n_points)
        st.pyplot(plotdists(sums))
        caption_figure('plots')
       

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

        cols=st.columns(2)
        resolution = cols[1].slider('resolution', 10, 100, 12)
        n_samples = cols[0].slider('n_points', 10, 100, 12)
        fig = chi2_demo(resolution=resolution, n_samples=n_samples)
        
        st.pyplot(fig)


    
    cols = st.columns(2)

    ezfuncs = {
        'linear' : {'func' : lambda x, a, b : a*x**1+b,
                    'p0' : np.array([1.5,2.7]),
                    'ptrue' : np.array([-2,10])},
        'parabolic' : {'func' :lambda x, a, b : a*x**2+b,
                    'p0' : np.array([-3,6]),
                    'ptrue' : [3,5]},
        'poly3' : {'func' : lambda x, a, b, c: a*x**3 + b*x**2 + c*x,
                    'p0' : np.array([1,2,5]),
                    'ptrue' : np.array([5,2,-4])},
        'sine' : {'func' : lambda x, a, b: a*np.sin(b+x),
                    'p0' : np.array([1.1,1.5]),
                    'ptrue' : np.array([1,1])},
        }
    f_dict = ezfuncs[cols[0].radio('function', ezfuncs.keys())]
    f = f_dict['func']
    p0 = f_dict['p0']
    p_true = f_dict['ptrue']

    n_samples = cols[0].slider('n_samples', 3,25,10)
    noise_scale = cols[0].slider('noise_scale', 0.0,1.,.2)
    fig = chi2_demo_2(f,p_true, p0, n_samples, noise_scale,
                    h=0.1, lr = 0.1, tol=.1, max_fev=500)
    cols[1].pyplot(fig)
    with st.expander('Links', expanded=True):
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
    text_dict2 = getText_prep(filename = text_path+'week3.md', split_level = 2)
     
    #st.header('Week 3')
    with st.expander('Week 3 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])
    st.markdown(text_dict['Header 2'])
    cols = st.columns(2)
    cols[0].markdown(text_dict['Header 3'])
    cols[1].markdown(text_dict['Header 4'])
    
    cols = st.columns(2)
    cols[0].markdown(text_dict['Header 5'])
    cols[1].markdown(text_dict2['Header 6'])
    
    st.markdown(text_dict['Header 7'])

    def sphereINcube_demo():
        # accept, reject to get pi
        cols = st.columns(2)


        # inputs
        n_points = cols[0].select_slider('Number of points', np.logspace(1,14,14,base=2, dtype=int))
        
        
        n_dim = cols[0].select_slider('Number of dimensions', np.arange(2,10,1, dtype=int))
        cols[0].markdown('We can kinda see how 3 dim. is showing a ball with air trapped in the corners.')
        p_norm = cols[0].slider('p (for norm)',0.,10.,2.)
        cols[0].markdown(r"""
        $$
            |x| = \left(\sum_i x_i^p\right)^{1/p}
        $$
        """)

        # make vecs and check norms
        X = np.random.uniform(-1,1,(n_points, n_dim))
        fig, ax = plt.subplots(figsize=(5,5))
        norm = np.sum(abs(X)**p_norm, axis=1)

        # plotting
        colors = [{0 : 'gold', 1 : 'green'}[n<=1] for n in norm]
        ax.scatter(X[:,0], X[:,1], c=colors,  norm = np.sum(X**2, axis=1)**.5, cmap='winter', alpha=.8)
        extent = 1.1 ; ax.set(xlim=(-extent,extent), ylim=(-extent,extent))
        
        # output
        cols[1].pyplot(fig)
        percentage = sum(abs(norm)<1)/n_points
        cols[0].write('Percentage inside the unit hypersphere = {:0.2f} giving us $\pi = {:0.4f}$'.format(percentage, percentage*4))

    sphereINcube_demo()
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
    'week 3': week3,
    #'week 4': week4,
    #'week 5': week5,
    #'week 6': week6,
    #'week 7': week7,
    'Utils explorer' : utils_explorer,
    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()