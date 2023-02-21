from utils.utils_global import *
from utils.utils_appstat import *


st.set_page_config(page_title="Applied statistics", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)


set_rcParams(style_dict = {
        'patch.facecolor' : (0.4, 0.065, 0.03),
        'axes.facecolor' : (0.04, 0.065, 0.03),
        'figure.facecolor' : (0.04, 0.065, 0.03),
        'xtick.color' : 'white',
        'ytick.color' : 'white',
        'text.color' : 'lightgreen',
        # 'axes.grid' : True,  # should we?,
        'figure.autolayout' : True,  # 'tight_layout',
        'axes.labelcolor' : "lightgreen",


    })


def home():
    st.title('Applied statistics')
    st.image('assets/images/Stable_diffusion__Mathematician_discovering_Neptune.png', width=420)
    st.caption('*Mathematician discovering Neptune* [[AI-generated with stable diffusion]](https://huggingface.co/spaces/stabilityai/stable-diffusion).')
    st.markdown("""
        Course taught by: Troels C. Petersen. 

        [course website](https://www.nbi.dk/~petersen/Teaching/AppliedStatistics2022.html)""")
    from streamlit.components.v1 import html
    my_html = """<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1399648774&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/antartica_traffic_control" title="Goblin Mode" target="_blank" style="color: #cccccc; text-decoration: none;">Goblin Mode</a> · <a href="https://soundcloud.com/antartica_traffic_control/other-eyes" title="Other Eyes" target="_blank" style="color: #cccccc; text-decoration: none;">Other Eyes</a></div>"""


def week1():
    
    text_dict = getText_prep(filename = text_path+'week1.txt', split_level = 1)
    #text_dict
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
    text_dict = getText_prep(filename = text_path+'week2.txt', split_level = 1)
    text_dict3 = getText_prep(filename = text_path+'week2.txt', split_level = 3)

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
    

    st.title('Simulation and More Fitting')
    text_dict = getText_prep_new(filename = text_path+'week3.txt')
    #st.header('Week 3')
    with st.expander('Week 3 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Intro'])
    st.markdown(text_dict['Transformation method'])
    with st.expander('Example', expanded=False):    
        st.markdown(text_dict['Header 4'])
    
    
    st.markdown(text_dict['Accept-reject'])
    with st.expander('Accept-reject code', expanded=False): 
        st.code(text_dict['Accept-reject code'])
    
    st.markdown(text_dict['Header 7'])


def week4():
    #st.header('Week 4')
    text_dict = getText_prep_new(filename = text_path+'week4.txt')
     
    with st.expander('Week 4 description', expanded=False):
        st.markdown(text_dict['description'])
    st.markdown(text_dict['Header 1'])

    def oneSampleZtest_DEMO():
        cols = st.columns(2)
        cols[0].markdown(text_dict['One Sample Z Test'])
        def randoms(size=100, dist = 'normal', mu=0, sigma=1):
            if dist=='normal':
                x = np.random.normal(mu, sigma, size)
            
            
            return x

        mu0 = 100
        sigma = 20
        n = cols[1].slider('num samples', 1,100,14)
        x = randoms(size=n, dist = 'normal', mu=110, sigma=sigma)
        fig, ax = plt.subplots()
        ax.hist(x)
        plt.grid()
        plt.close()
        cols[1].pyplot(fig)
        cols[1].image('https://www.z-table.com/uploads/2/1/7/9/21795380/8573955.png?759')

        
        Z = (np.mean(x)-mu0) / (sigma/n**.5) 
        cols[0].write(f'$Z= {round(Z,4)}$')
    with st.expander('One sample Z test', expanded=False):
        oneSampleZtest_DEMO()

    st.markdown(text_dict['Header 2'])


def week5():
    st.title('Bayes and MVA')
     
    text_dict = getText_prep_new(filename = text_path+'week5.txt')
    with st.expander('Week 5 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Bayes theorem and Baysian statistics'])

    st.markdown(text_dict['Multi-Variate Analysis (MVA)'])

    cols = st.columns(2)
    cols[0].markdown(text_dict['The linear Fisher discriminant'])

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    df = sns.load_dataset('iris')
    #df
    X = df.values[:,:-1]
    mapper = {
        'versicolor' : 0,
        'setosa' : 0,
        'virginica' : 1
    }
    y = np.array([mapper[i] for i in df.values[:,-1]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X = pca.fit_transform(X_scaled)

    print(pca.explained_variance_ratio_)
    fig = plt.figure()
    plt.title('True grouping')
    plt.scatter(X[:,0],X[:,1], c=y, alpha=.8, cmap='RdBu')
    plt.close()
    cols[1].pyplot(fig)
    
    mu0 = np.mean(X[y==0], axis=0)
    mu1 = np.mean(X[y==1], axis=0)
    #cols[1].markdown(f'mu0 = {mu0}, mu1 = {mu1}')
    
    S0 = np.cov(X[y==0].T)
    S1 = np.cov(X[y==1].T)
    #cols[1].write(f'S0 = {S0}, S1 = {S1}')

    n=len(X)
    S = 1/n * (S0+S1)
    #cols[1].write(f'S = {S}')

    w = mu0-mu1 * np.linalg.inv(S)
    #cols[1].write(f'w = {w}')

    new = w.T @ np.array([3,-1])
    #cols[1].write(f'new observation = {new}')
    y_pred = np.argmax(w.T@X.T, axis=0)
    #y_pred
    fig = plt.figure()
    plt.title('Predicted grouping by Fisher discriminant')
    plt.scatter(X[:,0],X[:,1], c=y_pred, alpha=.8, cmap='RdBu')
    plt.close()
    cols[1].pyplot(fig)


def week6():
    st.title('Machine learning and time series')

    text_dict = getText_prep_new(filename = text_path+'week6.txt' )
    with st.expander('Week 6 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    def Kmeans_knn_DEMO():
        X = makeBlobs(100)

    
        kMeans_labels, nitr = kMeans(X, nclusters=4)
        
        # classification
        y = kMeans_labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # test train split

        
        
        knn_labels = kNN(X_test, X_train, y_train, k = 4)
        fig, ax = plt.subplots(1,2, figsize=(7,3))
        
        ax[0].set(title=f'kMeans clustering, nitr={nitr}', xticks=[], yticks=[])
        ax[0].scatter(X[:,0], X[:,1], marker='x' ,c=kMeans_labels)


        ax[1].set(title='kNN classification', xticks=[], yticks=[])
        ax[1].scatter(X_train[:,0], X_train[:,1], c=y_train)
        ax[1].scatter(X_test[:,0], X_test[:,1],marker='x' , c=knn_labels)


        st.pyplot(fig)
        plt.close()
        st.caption('Demonstrations of clustering and classification algos. Crosses indicate unknown labels, circles have known labels')

    Kmeans_knn_DEMO()

    '#### Neural netorks'
    cols = st.columns(2)
    cols[1].image('https://www.investopedia.com/thmb/PgHPmalVUUHIQrp616mTdlmyD0I=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/dotdash_Final_Neural_Network_Apr_2020-01-5f4088dfda4c49d99a4d927c9a3a5ba0.jpg')
    cols[0].markdown('blank')

    '#### Decision Trees'
    cols = st.columns(2)
    cols[1].image('https://s3.ap-south-1.amazonaws.com/techleer/252.jpg')
    cols[0].markdown('Consecutive binary questions, allow us to carve out straight-walled sections of the phase space.')

    '---'
    '#### Timeseries'
    cols = st.columns(2)
    cols[1].image('https://miro.medium.com/max/1100/1*r6F3L0fG1VkH9zz5uZjT7w.webp')
    cols[0].markdown('')


def week7():
    st.title('Advanced fitting & Calibration')
    
    text_dict = getText_prep(filename = text_path+'week7.txt', split_level = 1)
    
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
    'Simulation and Fitting': week3,
    'Hypothesis testing': week4,
    'Bayes and MVA': week5,
    'Machine learning and time series': week6,
    'week 7': week7,
    'Utils explorer' : utils_explorer,
    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()


