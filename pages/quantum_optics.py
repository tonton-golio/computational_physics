from utils.utils_qopt import *
import sys


st.set_page_config(page_title="Quantum Optics", 
    page_icon="☯️ ", 
    layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
        #'Get Help': 'https://github.com/tonton-golio/computational_physics',
        #'Report a bug': "https://github.com/tonton-golio/computational_physics",
        'About': "Writer: Yoshiaki Horiike (wnq616)"
        })



plt.rcdefaults()
#plt.style.use('dark_background')
#set_rcParams()

# pages
def home():
    text_dict = getText_prep(filename = text_path+'home.md', split_level = 1)

    st.header('Welcome to Quantum Optics!')

    name = '__Contents__'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])

    name = '__Literature__'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])
        
    name = '__Lecturers__'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])

    st.markdown("From [kurser.ku.dk](https://kurser.ku.dk/course/nfyk13006u/2022-2023)")

    name = '__Other useful webpage for quantum optics__'
    with st.expander(name, expanded=True):
        st.markdown(text_dict[name])


def topic1():
    text_dict = getText_prep(filename = text_path+'topic1.md', split_level = 1)

    st.header('Quantization of the free electromagnetic field I')
    st.write('8 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic2():
    text_dict = getText_prep(filename = text_path+'topic2.md', split_level = 1)

    st.header('Quantization of the free electromagnetic field II')
    st.write('13 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic3():
    text_dict = getText_prep(filename = text_path+'topic3.md', split_level = 1)

    st.header('Coherent states')
    st.write('15 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic4():
    text_dict = getText_prep(filename = text_path+'topic4.md', split_level = 1)

    st.header('Density operator and phase-space distributions')
    st.write('20 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

    st.header('Coherent state on phase space')
    plot_coherent_on_phase_space()

    st.header('Number state on phase space')
    plot_number_on_phase_space()


def wigner_function_gallery():
    st.header('The Wigner function gallery')

    st.subheader('Coherent state')
    plot_wigner_coherent()

    st.subheader('Number state (Fock state)')
    plot_wigner_number()

    st.subheader('Schrödinger cat state')
    plot_wigner_cat()

def wigner_function_gallery_3d():
    st.header('The Wigner function gallery 3D')

    st.subheader('Coherent state')
    plot_wigner_coherent(three_dimensional=True)

    st.subheader('Number state (Fock state)')
    plot_wigner_number(three_dimensional=True)

    st.subheader('Schrödinger cat state')
    plot_wigner_cat(three_dimensional=True)

def cool_wigner_papers():
    st.header("Cool Wigner function papers")

    text_dict = getText_prep(filename = text_path+'papers.md', split_level = 1)
    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=True):
            st.markdown(text_dict[name])


# Navigator
topic_dict = {
    'Welcome!': home,
    'Topic 1': topic1,
    'Topic 2': topic2,
    'Topic 3': topic3,
    'Topic 4': topic4,
    #'Topic 5': topic5,
    #'Topic 6': topic6,
    #'Topic 7': topic7,
    #'Topic 8': topic8,
    #'Topic 9': topic9,
    #'Topic 10': topic10,
    #'Topic 11': topic11,
    #'Topic 12': topic12,
    #'Topic 13': topic13,
    #'Topic 14': topic14,
    'The Wigner function gallery': wigner_function_gallery,
    'The Wigner function gallery 3D': wigner_function_gallery_3d,
    'Cool Wigner function papers': cool_wigner_papers,
    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()
