from utils.utils_dynbio import *

def home():
    st.title('Dynamical Models in Molecular Biology')
    #st.image('assets/images/Stable_diffusion__Mathematician_discovering_Neptune.png', width=420)
    #st.caption('Stable diffusion response to prompt: *Mathematician discovering Neptune*.')

def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
     
    #st.header('Week 3')
    with st.expander('Week 1 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def week2():
    text_dict = getText_prep(filename = text_path+'week2.md', split_level = 1)
     
    st.header('Regulation of gene expression')

    name = 'Week 2 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Differential equation for creation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Differential equation for degradation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Differential equation for creation and degradation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Differential equation for transcription and translation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Number of molecules vs concentration of molecules'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Transcriptional regulation: Repression'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])


    name = 'Transcriptional regulation: Activation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    threshold = cols[0].slider('Dissociation constant K', 0.0, 2.0, 1.0)
    coeff = cols[1].slider('Hill coefficient H', 1, 10, 1)

    fig = plot_hill_function(threshold, coeff, activation=False)
    cols[0].pyplot(fig)

    fig = plot_hill_function(threshold, coeff, activation=True)
    cols[1].pyplot(fig)

    name = 'Transcriptional regulation: sRNA'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    #fig = srna_simulation()
    #st.pyplot(fig)

def week3():
    text_dict = getText_prep(filename = text_path+'week3.md', split_level = 1)
     
    st.header('Mutational Analysis')

    name = 'Week 3 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'What causes mutations?'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Fidelity in DNA replication and gene expression'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Proofreading in DNA replication'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Recombination'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Mutants'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Distribution for mutation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Binomial distribution'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    N = cols[0].slider('Number of trials N', 0, 100, 10)
    p = cols[1].slider('Probability of finding mutated cell p', 0.0, 1.0, 1.0/6.0)
    k = np.arange(21)
    y = factorial(N) * p**k * (1.0-p)**(N-k) / (factorial(N-k)*factorial(k)) 
    fig, ax = plt.subplots()
    ax.plot(k, y)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_title("Binomial distribution")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P_N(k)$")
    st.pyplot(fig)

    name = 'Poisson distribution'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    m = st.slider('m', 0, 32, 1)
    k = np.arange(21)
    y = factorial(N) * p**k * (1.0-p)**(N-k) / (factorial(N-k)*factorial(k)) 
    fig, ax = plt.subplots()
    ax.plot(k, y)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_title("Binomial distribution")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P_N(k)$")
    st.pyplot(fig)


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
    #'Welcome': home,
    #'week 1': week1,
    'week 2': week2,
    'week 3': week3,
    #'week 4': week4,
    #'week 5': week5,
    #'week 6': week6,
    #'week 7': week7,    
    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()
