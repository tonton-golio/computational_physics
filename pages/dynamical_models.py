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

    threshold = 1
    coeff = 2
    x = np.linspace(0, threshold*2, 1000)
    y = 1.0 / (1.0 + (x/threshold)**coeff)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(0.5, c='k', ls='--')
    ax.axvline(threshold, c='k', ls='--')
    ax.set_title("Hill function for repression, coefficient={}".format(coeff))
    ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
    ax.set_ylabel("Value of Hill function")
    st.pyplot(fig)

    threshold = 1
    coeff = 2
    x = np.linspace(0, threshold*2, 1000)
    y = (x/threshold)**coeff / (1.0 + (x/threshold)**coeff)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(0.5, c='k', ls='--')
    ax.axvline(threshold, c='k', ls='--')
    ax.set_title("Hill function for activation, coefficient={}".format(coeff))
    ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
    ax.set_ylabel("Value of Hill function")
    st.pyplot(fig)

    name = 'Transcriptional regulation: Activation'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

    name = 'Transcriptional regulation: sRNA'
    with st.expander(name, expanded=False):
        st.markdown(text_dict[name])

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
    'Welcome': home,
    'week 1': week1,
    'week 2': week2,
    'week 3': week3,
    'week 4': week4,
    'week 5': week5,
    'week 6': week6,
    'week 7': week7,    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()
