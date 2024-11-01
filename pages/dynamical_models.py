from utils.utils_dynbio import *

st.set_page_config(page_title="Dynamical Models", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="expanded", 
    menu_items=None)


plt.rcdefaults()
#set_rcParams()

# pages
def home():
    text_dict = getText_prep(filename = text_path+'home.md', split_level = 1)

    st.header('Dynamical Models in Molecular Biology')

    name = 'Welcome to DynBio!'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])

    name = 'Aim of the course: BRIDGE THE GAP'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])

def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)

    st.header('Quantifying Noise in Gene Expression')

    name = 'Week 1 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Cellular identity'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'What is gene expression?'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Transcriptional regulation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'How to measure gene expression?'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Two way of measuring gene expression - single cell and bulk'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Why is the gene expression noisy?'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Molecules in the cell move via diffusion'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Definition of the total noise'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Extrinsic noise'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Intrinsic noise'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Decomposing total noise into intrinsic noise and extrinsic noise'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(4)
    kmRNA = cols[0].number_input('kmRNA', min_value=0.0, max_value=None, value=10.0)
    gmRNA = cols[0].number_input('gmRNA', min_value=0.0, max_value=None, value=1.0)
    kpro  = cols[1].number_input('kpro',  min_value=0.0, max_value=None, value=1.0)
    gpro  = cols[1].number_input('gpro',  min_value=0.0, max_value=None, value=0.1)
    Ngene = cols[2].number_input('Ngene', min_value=1, max_value=2, value=1)
    Ncell = cols[2].number_input('Ncell', min_value=1, max_value=2, value=1)
    NRepressor = cols[3].number_input('NRepressor', min_value=0, max_value=None, value=0)
    parameters = [kmRNA, gmRNA, kpro, gpro, Ngene, Ncell, NRepressor]

    results, fig = plot_noise(*parameters)
    st.write("mRNA average: ",               results[0])
    st.write("protein average: ",            results[1])
    st.write("protein variance: ",           results[2])
    st.write("protein standard deviation: ", results[3])
    st.write("protein total noise: ",        results[4])
    if len(results) == 7:
        st.write("Extrinsic noise: ", results[5])
        st.write("Intrinsic noise: ", results[6])
    st.pyplot(fig)

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
    N1 = cols[0].slider('Number of trials N1', 1, 50, 10)
    p1 = cols[0].slider('Probability of finding mutated cell p1', 0.0, 1.0, 1.0/6.0)
    N2 = cols[1].slider('Number of trials N2', 1, 50, 20)
    p2 = cols[1].slider('Probability of finding mutated cell p2', 0.0, 1.0, 2.0/6.0)
    fig, ax = plot_binomial(N1, p1, N2, p2)
    st.pyplot(fig)

    name = 'Poisson distribution'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    m1 = cols[0].slider('Average number of mutation m1', 1, 32, 1)
    m2 = cols[1].slider('Average number of mutation m2', 1, 32, 10)
    fig = plot_poisson(m1, m2)
    st.pyplot(fig)

    name = 'Binomial vs Poisson'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    m = cols[1].slider('Average number of mutation m', 1, 100, 8)
    N = cols[0].slider('Number of trials N', m, 100, m)
    fig = plot_binomial_poisson(N, m)
    st.pyplot(fig)

def week4():
    text_dict = getText_prep(filename = text_path+'week4.md', split_level = 1)
    
    st.header('Bactearial Growth Physiology')
     
    name = 'Week 4 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Phase of bacterial growth'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Difinition of steady-state growth'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Bacterial growth las by Jacques Monod (1949)'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    lambda_max = cols[0].slider('lambda_max', 0.0, 2.0, 1.25)
    K_S = cols[1].slider('K_S', 0.0, 10.0, 0.5)
    fig, ax = plot_michaelis_menten1(lambda_max, K_S)
    st.pyplot(fig)

    name = 'Frederick C. Neidhardt (1999)'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Bacterial biomass is mainly protein'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

def week5():
    text_dict = getText_prep(filename = text_path+'week5.md', split_level = 1)
    
    st.header('Gene Regulatory Networks')
     
    name = 'Week 5 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Type of regulation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Statistics of regulatory function'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Real network and how we understand gene regulation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Type of network motif'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'How can we find positive feedback and negative feedback'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Biological example of positive/negative regulation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Biological example of feed-forward loops'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Simplification of dynamical equation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Obtaining steady state concentration from graph'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    H = cols[0].slider('Hill coefficient H', 1, 10, 1, key=0)
    gamma_P = cols[1].slider('Gamma_P', 0.0, 1.0, 0.5, key=1)
    fig, ax = plot_solve_regulation(H, gamma_P)
    st.pyplot(fig)

    name = 'How about negative regulation?'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    cols = st.columns(2)
    H = cols[0].slider('Hill coefficient H', 1, 10, 1, key=2)
    gamma_P = cols[1].slider('Gamma_P', 0.0, 1.0, 0.5, key=3)
    fig, ax = plot_solve_regulation(H, gamma_P, positive=False)
    st.pyplot(fig)

def week6():
    text_dict = getText_prep(filename = text_path+'week6.md', split_level = 1)
    
    st.header('Signal transduction')

    name = 'Week 6 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'Beginning of signal transduction'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'Signla transduction involves many steps'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'A well studied example of sinal transduction'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'How tumbling frequency depends on ligand concentraion'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Chemotaxis requires adaptation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'Modeling chemotaxis'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'Signal transduction in space'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
     
    name = 'Cell-to-cell communication'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])
              
def week7():
    text_dict = getText_prep(filename = text_path+'week5.md', split_level = 1)
    
    st.header('Bactearial Growth Physiology')
     
    name = 'Week 4 description'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])


# Navigator
topic_dict = {
    'Welcome': home,
    'week 1': week1,
    'week 2': week2,
    'week 3': week3,
    'week 4': week4,
    'week 5': week5,
    'week 6': week6,
    #'week 7': week7,    
    }

# run with analytics
# streamlit_analytics.start_tracking()

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))
run_topic = topic_dict[topic] ; run_topic()

# streamlit_analytics.stop_tracking()
