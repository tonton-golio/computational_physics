from utils.utils_appstat import *


def home():
    st.title('Applied statistics')

def week1():
    st.title('Week 1')
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
     
    #st.header('Week 1')
    with st.expander('Week 1 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    with st.expander('Mean & Standard Deviation', expanded=False):
        st.markdown(text_dict['Mean'])
        
        mean_picker = st.selectbox('There are different measures hereof',['Geometric mean', 'Arithmetic mean', 'Median',  'Mode', 'Harmonic', 'Truncated'])

        cols = st.columns(2)
        cols[0].markdown(text_dict[mean_picker])
        cols[1].markdown(text_dict[mean_picker+' code'])
        
        '##### _____________________________________________________________________________________________'
        
        cols = st.columns(4)
        dists = 'n_normal, n_exp, n_cauchy, truncate'.split(', ')
        
        n_normal, n_exp, n_cauchy, truncate = (cols[i].slider(name, 0, 100, 1000) for i, name in enumerate(dists))
        
        normal = np.random.randn(n_normal)
        exp = np.random.exponential(n_exp) 
        cauchy = np.random.standard_cauchy(n_cauchy)
        
        arr = np.hstack([normal, exp, cauchy])

        fig = showMeans(arr)   # Perhaps change this to 2D
        st.pyplot(fig)


    #with st.expander('Standard Deviation', expanded=False):
        st.markdown(text_dict['STD'])

        # compare N-1 and N...
        #if st.button('run std_calculations'):
        st.pyplot(std_calculations())


        st.markdown(text_dict['Weighted mean'])


    with st.expander('Correlation', expanded=False):
        

        st.markdown(text_dict['Correlation'])


    with st.expander('Central limit theorem', expanded=False):
        st.markdown(text_dict['Central limit theorem'])
        

        cols = st.columns(2)
        # roll a die
        cols[0].pyplot(roll_a_die(420))

        # roll dice
        cols[1].pyplot(roll_dice())



        cols = st.columns(2)

        cols[0].markdown(text_dict['Central limit theorem 2'])


        
        n_points = cols[1].slider('n_points',10,2000, 100)
        x = np.random.randn(n_points)
        std = np.std(x)
        mean = np.mean(x)
        fig, ax = plt.subplots(figsize=(3,3))
        ax.hist(x, bins=42)
        for n in range(-2,3):
            ax.axvline(mean+n*std, c='r', ls='--')
        tick_locs = [mean+n*std for n in range(-2,3)]
        tick_vals = [f'{n}$\sigma$' for n in range(-2,3)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_vals)
        plt.close()
        cols[1].pyplot(fig)

        x -= mean
        sigs = [len(x[abs(x)<mean+n*std])/n_points for n in range(1,5)]
        sigs = np.round(sigs, 3)*100
        cols[0].markdown(f'''
            |num. $\sigma$   | percentage inside theoretical  | percentage inside {n_points} samples |
            |---|---|---|
            |1   |68   | {sigs[0]} |
            |2   |95   | {sigs[1]} |
            |3   |99.7   | {sigs[2]} |
            |4   |99.99995   | {sigs[3]} |
        ''')

        st.markdown(text_dict['Central limit theorem 3'])
    
    with st.expander('Error propagation', expanded=True):
        st.markdown(text_dict['Error propagation'])

    st.markdown(text_dict['Estimating uncertainties'])
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


    st.markdown(text_dict['Links'])
    
def week2():
    text_dict = getText_prep(filename = text_path+'week2.md', split_level = 1)
     
    #st.header('Week 2')
    with st.expander('Week 2 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    st.pyplot(PDFs(1000))

    st.markdown(text_dict['Header 2'])

    st.pyplot(fitSimple(size =  100, nsteps = 100))

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


# Navigator
topic_dict = {
    'week 1': week1,
    'week 2': week2,
    'week 3': week3,
    'week 4': week4,
    'week 5': week5,
    'week 6': week6,
    'week 7': week7,    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()



