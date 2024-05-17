from utils.utils_qopt import *
import sys


st.set_page_config(page_title="Quantum Optics", 
    page_icon="☯️ ", 
    layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
        'About': "Writer: Yoshiaki Horiike (wnq616)"
        })



# pages
def wigner_function_gallery():
    st.header('The Wigner function gallery')
    st.markdown("$W(q, p) = \\frac{1}{2\\pi\\hbar} \\int_{-\\infty}^\\infty \\Braket{q+\\frac{1}{2}x|\\hat{\\rho}|q-\\frac{1}{2}x} e^{ipx/\\hbar} \\mathrm{d}x$")
    st.markdown('---')
    theme = st.radio("Choose a theme", ("Light", "Dark"), horizontal=True)
    st.markdown('---')

    if theme=="Light":
        plt.rcdefaults()
    else:
        plt.style.use('dark_background')

    st.subheader('Number state (Fock state)')
    plot_wigner_number(theme)

    st.subheader('Coherent state')
    plot_wigner_coherent(theme)

    st.subheader('Squeezed state')
    plot_wigner_squeezed(theme)

    st.subheader('Schrödinger cat state')
    plot_wigner_cat(theme)

def wigner_function_gallery_3d(theme='Light'):
    st.header('The Wigner function gallery 3D')
    st.markdown("$W(q, p) = \\frac{1}{2\\pi\\hbar} \\int_{-\\infty}^\\infty \\Braket{q+\\frac{1}{2}x|\\hat{\\rho}|q-\\frac{1}{2}x} e^{ipx/\\hbar} \\mathrm{d}x$")

    st.subheader('Number state (Fock state)')
    plot_wigner_number(theme, three_dimensional=True)

    st.subheader('Coherent state')
    plot_wigner_coherent(theme, three_dimensional=True)

    st.subheader('Squeezed state')
    plot_wigner_squeezed(theme, three_dimensional=True)

    st.subheader('Schrödinger cat state')
    plot_wigner_cat(theme, three_dimensional=True)

def cool_wigner_papers():
    st.header("Cool Wigner function papers")

    text_dict = getText_prep(filename = text_path+'papers.md', split_level = 1)
    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=True):
            st.markdown(text_dict[name])


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

    name = '__Writer__'
    with st.expander(name, expanded=True):
        st.markdown("Yoshiaki Horiike")

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

def topic5():
    text_dict = getText_prep(filename = text_path+'topic5.md', split_level = 1)

    st.header('Quantum coherence functions of the 1st order and photodetection')
    st.write('22 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic6():
    text_dict = getText_prep(filename = text_path+'topic6.md', split_level = 1)

    st.header('Quantum coherence functions of the 2nd order')
    st.write('27 Feb 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic7():
    text_dict = getText_prep(filename = text_path+'topic7.md', split_level = 1)

    st.header('Beam splitters')
    st.write('1 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic8():
    text_dict = getText_prep(filename = text_path+'topic8.md', split_level = 1)

    st.header('Interferometry with quantum fields')
    st.write('6 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic9():
    text_dict = getText_prep(filename = text_path+'topic9.md', split_level = 1)

    st.header('Squeezed light I')
    st.write('8 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic10():
    text_dict = getText_prep(filename = text_path+'topic10.md', split_level = 1)

    st.header('Squeezed light II')
    st.write('13 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic11():
    text_dict = getText_prep(filename = text_path+'topic11.md', split_level = 1)

    st.header('Quantum teleportation')
    st.write('15 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic12():
    text_dict = getText_prep(filename = text_path+'topic12.md', split_level = 1)

    st.header('Atom field interaction I')
    st.write('20 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic13():
    text_dict = getText_prep(filename = text_path+'topic13.md', split_level = 1)

    st.header('Atom field interaction II')
    st.write('22 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])

def topic14():
    text_dict = getText_prep(filename = text_path+'topic14.md', split_level = 1)

    st.header('Experimental cavity QED')
    st.write('27 Mar 2023')

    for key in text_dict.keys():
        name = key
        with st.expander(name, expanded=False):
            st.markdown(text_dict[name])


# Navigator
topic_dict = {
    'The Wigner function gallery': wigner_function_gallery,
    'The Wigner function gallery 3D': wigner_function_gallery_3d,
    'Cool Wigner function papers': cool_wigner_papers,
    'Welcome!': home,
    'Topic 1': topic1,
    'Topic 2': topic2,
    'Topic 3': topic3,
    'Topic 4': topic4,
    'Topic 5': topic5,
    'Topic 6': topic6,
    'Topic 7': topic7,
    'Topic 8': topic8,
    'Topic 9': topic9,
    'Topic 10': topic10,
    'Topic 11': topic11,
    'Topic 12': topic12,
    'Topic 13': topic13,
    'Topic 14': topic14,
    }

# run with analytics 
streamlit_analytics.start_tracking()

topic = st.sidebar.selectbox("Select a topic!" , list(topic_dict.keys()))
run_topic = topic_dict[topic] ; run_topic()

streamlit_analytics.stop_tracking()
