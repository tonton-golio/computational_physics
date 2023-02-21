from utils.utils_continuum import *

set_rcParams()

def frontpage():    
    text_dict = getText_prep(textfile_path+'frontpagetext.md', split_level = 2)
    st.markdown(r"""# __Continuum Mechanics__""")
    
    cols = st.columns([2,1])
    cols[0].write("""
    GitHub: https://github.com/tonton-golio/computational_physics.git
    
    Welcome to the student-run collaboratory OpenSource Streamlit page for the 
    course _Continuum Mechanics_, here we have collected notes, scripts, tips 
    and tricks to help out fellow students in the course.
    Streamlit is a tool for hosting a free website, displaying the contents of 
    a GitHub repository. It comes with convenient commands, that allows an easy 
    integration of LaTeX and Python into one interactive platform. Topics may 
    be selected on the left of this page.
    
    __Table of contents__
    1. Frontpage
    2. The Continuum Approximation
    3. Tensor Fundamentals
    4. Elasticity (*TBC*)
    5. Fluid Fundamentals
    6. Fluids in Motion (*TBC*)
    
    __Simulators__
    1. Iceberg Simulator (*TBS*)
    2. Strain and Stress Visualizer (*TBS*)
    3. Finite Element Modelling (*TBS*)
    4. Useful Python Packages (*TBS*)
    """)
#   To be started
#   Iceberger widget
#   Useful integrals and fundamental field theory   
#   Useful python packages

    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols[0].markdown(text_dict[key])

    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Airplane_vortex_edit.jpg/330px-Airplane_vortex_edit.jpg',width=300)
    cols[1].caption('Vortex displayed by dye caught in the turbulence of an landing airplane.')
    cols[1].image('https://blogs.egu.eu/divisions/gd/files/2021/10/tectonic_modeling-1400x800.png',width=300)
    cols[1].caption('Finite Element Modelling of Plate Tectonics. Coloration displays stress.')
    cols[1].image('https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/tensor-calculus-doug-morgan.jpg',width=390)
    cols[1].caption('Tensor Calculus is a piece of digital artwork by Doug Morgan')
    cols[1].image('https://legacy.altinget.dk/images/article/257037/84969.jpg',width=300)
    cols[1].caption('Danish Fregatte HDMS TRITON on a patrol, clearing a path in the ice shelves into Kangerlussuaq for civian shipping.')
    st.write(signatur)

def contapprox():
    st.markdown(r"""# The Continuum Approximation""")
    text_dict = getText_prep(filename = textfile_path+'contapprox.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict[key])
        cols[1].image('https://www.campoly.com/files/8214/6721/8144/2014-08-variable-pressure-scanning-electron.jpg')
        cols[1].caption('Scanning electron microscopy micrograph of hydrogel pore structure taken in a hydrated state\ https://www.campoly.com/blog/2014-08-variable-pressure-scanning-electron/')

    key="__Density Fluctuations__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        st.image(imagefile_path+'densityfluctuations.png')
        st.caption('From Physics of Continuous Matter 2nd edition by Benny Lautrup')
        
    key="__Macroscopic Smoothness__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key ="__Velocity Fluctuations__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
            
    key ="__Mean-Free-Path__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
    st.write(signatur)

def tensorfundamentals():
    st.markdown(r"""# __Tensor Fundamentals__""")
    text_dict = getText_prep(filename = textfile_path+'tensorfundamentals.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict[key])
        cols[1].video('https://www.youtube.com/watch?v=7HikEFicLO8')
        cols[1].caption('Introduction to rank 2 tensors, including gradient/velocity tensor, strain and stress tensor and spin/rotation tensor. All is used throughout the course.')

    key="__Cauchy Stress Tensor__"
    with st.expander(key, expanded=False):
        st.image('https://upload.wikimedia.org/wikipedia/commons/b/b3/Components_stress_tensor_cartesian.svg',width=500)
        st.caption('Illustration of the pressures and directions that the elements of $\sigma_{ij}$ represents.')
        st.markdown(text_dict[key])

    key="__Stress Deviator and Invariants__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
    
    key="__Cauchy Strain Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key="__Gradient and Spin Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
    st.write(signatur)
        
def elasticity():
    st.markdown(r"""# __Elasticity__""")
    text_dict = getText_prep(filename = textfile_path+'elasticity.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        cols = st.columns([1,1])
        cols[0].markdown(text_dict[key])
        cols[1].image('https://www.campoly.com/files/8214/6721/8144/2014-08-variable-pressure-scanning-electron.jpg')
        cols[1].caption('Scanning electron microscopy micrograph of hydrogel pore structure taken in a hydrated state')
        
    key='__Work__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key='__Hookes Law__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    st.write(signatur)

def fluidfundamental():
    st.markdown(r"""# __Fluid Fundamentals__""")
    text_dict = getText_prep(filename = textfile_path+'fluids.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        #cols = st.columns([1,1])
        st.markdown(text_dict[key])
        #cols[1].video('https://www.youtube.com/watch?v=7HikEFicLO8')
        #cols[1].caption('Introduction to rank 2 tensors, including gradient/velocity tensor, strain and stress tensor and rotation tensor. All is used throughout the course.')

    key="__Buoyancy and Stability__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key="__Pressure__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key="__Equation of State__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
    key="__sample headline__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    st.write(signatur)
    
def fluidsinmotion():
    st.markdown(r"""# __Fluids in Motion__""")
    text_dict = getText_prep(filename = textfile_path+'fluids.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])
        
    st.write(signatur)

#Navigator
if __name__ == '__main__':
    functions = [frontpage, contapprox, tensorfundamentals, elasticity, fluidfundamental, fluidsinmotion]
    names = ['1. Frontpage', '2. The Continuum Approximation', '3. Tensor Fundamentals', '4. Elasticity', '5. Fluid Fundamentals','6. Fluids in Motion']
    navigator(functions, names)