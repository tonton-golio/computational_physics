from utils.utils_continuum import *

set_rcParams()

def frontpage():    
    text_dict1 = getText_prep(textfile_path+'frontpagetext.md', split_level = 2)
    #st.write(text_dict1.keys())
    st.title("""Continuum Mechanics""")
    cols = st.columns([2,1])
    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Airplane_vortex_edit.jpg/330px-Airplane_vortex_edit.jpg')
    cols[1].caption('Vortex around an airplane')
    
    cols[0].write("""
    __Table of contents__
    1. Frontpage
    2. The Continuum Approximation
    3. Stress and Strain
    4. Elasticity
    5. Fluids
    
    """)
#   To be started
#   Iceberger widget
#   Useful integrals    
#   Useful python packages

    
    cols = st.columns([2,1])
    
    key='Introduction'
    with st.expander(key, expanded=True):
        cols[0].markdown(text_dict1[key])

    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png')
    cols[1].caption('Amedeo Avogadro')


def contapprox():
    st.markdown(r"""# The Continuum Approximation""")
    text_dict2 = getText_prep(filename = textfile_path+'contapprox.md', split_level = 2)
    
    key='Introduction'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict2[key])
        cols[1].image('https://www.campoly.com/files/8214/6721/8144/2014-08-variable-pressure-scanning-electron.jpg')
        cols[1].caption('Scanning electron microscopy micrograph of hydrogel pore structure taken in a hydrated state')

    key="Density Fluctuations"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])
        
    key="Macroscopic Smoothness"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

    key ="Velocity Fluctuations"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])
            
    key ="Mean-Free-Path"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

def stressandstrain():
    st.markdown(r"""# __Stress and Strain__""")
    text_dict2 = getText_prep(filename = textfile_path+'stressandstrain.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict2[key])
        cols[1].video('https://www.youtube.com/watch?v=7HikEFicLO8')
        cols[1].caption('Introduction to rank 2 tensors, including gradient/velocity tensor, strain and stress tensor and rotation tensor. All is used throughout the course.')

    key="__Cauchy Stress Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

    key="__Stress Deviator__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])
        
    key="__Cauchy Strain Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

def fluids():
    st.markdown(r"""# __Fluids__""")
    text_dict2 = getText_prep(filename = textfile_path+'fluids.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        #cols = st.columns([1,1])
        st.markdown(text_dict2[key])
        #cols[1].video('https://www.youtube.com/watch?v=7HikEFicLO8')
        #cols[1].caption('Introduction to rank 2 tensors, including gradient/velocity tensor, strain and stress tensor and rotation tensor. All is used throughout the course.')

    key="__Buoyancy and Stability__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

    key="__Pressure__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])

    key="__Equation of State__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])
        
    key="__sample headline__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict2[key])
      
def elasticity():
    st.markdown(r"""# __Elasticity__""")
    text_dict = getText_prep(filename = textfile_path+'elasticity.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict[key])
        cols[1].image('https://www.campoly.com/files/8214/6721/8144/2014-08-variable-pressure-scanning-electron.jpg')
        cols[1].caption('Scanning electron microscopy micrograph of hydrogel pore structure taken in a hydrated state')
    
    key='__Hookes Law__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
#Navigator
if __name__ == '__main__':
    functions = [frontpage, contapprox, stressandstrain, fluids, elasticity]
    names = ['1. Frontpage', '2. The Continuum Approximation', '3. Stress and Strain', '4. Elasticity', '5. Fluids']
    navigator(functions, names)