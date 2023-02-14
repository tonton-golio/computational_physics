from utils.utils_continuum import *

set_rcParams()

def frontpage():
    text = getText_prep_new(textfile_path+'frontpagetext.md',3)
    st.markdown(text)
    cols = st.columns(2)
    cols[0].image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Airplane_vortex_edit.jpg/330px-Airplane_vortex_edit.jpg')
    cols[0].caption('Vortex around an airplane')

    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png')
    cols[1].caption('Amedeo Avogadro')

    st.title("""Table of contents""")
    st.write("""
    1. Frontpage
    2. The Continuum Approximation
    
    To do:
    3. Iceberger widget
    4. Stress and Strain page
    5. Useful integrals    
    """)

def contapprox():
    st.markdown(r"""# The Continuum Approximation""")
    text_dict = getText_prep(filename = textfile_path+'contapprox.md', split_level = 2)
    
    text_expander(key="Introduction", text_dict=text_dict, expanded=True)
    #text_expander(key="Introduction", text_dict=text_dict)

    key="Macroscopic Smoothness"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key ="Velocity Fluctuations"
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])
        
    key ="Mean-free-path"
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])

#Navigator
if __name__ == '__main__':
    functions = [frontpage, contapprox]
    names = ['Frontpage', 'The Continuum Approximation']
    navigator(functions, names)