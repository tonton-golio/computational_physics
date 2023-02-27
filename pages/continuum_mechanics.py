from utils.utils_continuum import *

set_rcParams()

def frontpage():    
    #Comments are given on the first topic page. Structural comments are omitted after.
    #Import text from assets directory defined in utils/utils_continuum.py with
    #getText_prep from utils/utils_globals.py
    text_dict = getText_prep(textfile_path+'frontpagetext.md', split_level = 2)
    #Write headline
    st.markdown(r"""# __Continuum Mechanics__""")
    
    #Define columns to structure text, make their relative size 2:1
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
    5. Finite Element Modelling
    6. Fluids at Rest
    7. Fluids in Motion (*TBC*)
    
    __Appendix__
    * Iceberg Simulator (*TBS*)
    * Stress and Strain Visualizer (*TBC*)
    * Finite Element Modelling (*TBS*)
    * Useful Python Packages
    """)
    
    #From Getfile_prep, keys mark the division between chapters in the 
    key='__Introduction__'
    #Input text in expandable tab on screen
    with st.expander(key, expanded=True):
        #Write text loaded from md file and imported with getfile_prep
        cols[0].markdown(text_dict[key])
    
    #Show images on Streamlit site
    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Airplane_vortex_edit.jpg/330px-Airplane_vortex_edit.jpg',width=300)
    cols[1].caption('Vortex displayed by dye caught in the turbulence of an landing airplane.')
    cols[1].image('https://blogs.egu.eu/divisions/gd/files/2021/10/tectonic_modeling-1400x800.png',width=300)
    cols[1].caption('Finite Element Modelling of Plate Tectonics. Coloration displays stress.')
    cols[1].image('https://legacy.altinget.dk/images/article/257037/84969.jpg',width=300)
    cols[1].caption('Danish Fregatte HDMS TRITON on a patrol, clearing a path in the ice shelves outside Kangerlussuaq for civilian shipping.')
    cols[1].image('https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/tensor-calculus-doug-morgan.jpg',width=300)
    cols[1].caption('Tensor Calculus is a piece of digital artwork by Doug Morgan.')
    
    #Contact info signature, editable in utils/utils_continuum.py
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
        st.image('https://images.squarespace-cdn.com/content/v1/5a33517f268b961f10de794c/1630622947614-L1I8T9MRPM7TOGUR3PMM/Cheese+Pull.jpg')
        st.caption('Cheese deformed into strings by applying a pulling pressure at a $45\deg$ angle. The eigenbasis of the stress tensor aligns with the pulling force along with a negative pressure perpendicular to the force. Strings form along the positive eigenvector and crevasses form perpendicular to the negative eigenvector, and depending on the cheese rheology enormous deformation is achievable.')
    
    key="__Cauchy Strain Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key="__Velocity Gradient and Spin Tensor__"
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
    st.write(signatur)
        
def elasticity():
    st.markdown(r"""# __Elasticity__""")
    text_dict = getText_prep(filename = textfile_path+'elasticity.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])
        st.image('https://nuclear-power.com/wp-content/uploads/2019/11/Hookes-law-stress-strain-curve.png',width=500)

    key='__Work__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key='__Hookes Law__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    st.write(signatur)
    
def finiteelementmodelling():
    st.markdown(r"""# __Finite Element Modelling__ """)
    text_dict = getText_prep(filename = textfile_path+'fem.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])
        
    key='__Weighted Residuals__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
    
    key='__Least Squares Method__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])
        
    key='__Collocation Method__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])

    key='__Galerkin\'s Method__'
    with st.expander(key, expanded=False):
        st.image('https://www.researchgate.net/profile/T-Zohdi/publication/266967028/figure/fig1/AS:668988438765580@1536510681302/Orthogonality-of-the-approximation-error.png',width=300)
        st.markdown(text_dict[key])    
    
    st.write(signatur)

def fluidfundamental():
    st.markdown(r"""# __Fluids at Rest__""")
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

def stressandstrainvisualizer():
    st.markdown(r"""# __Stress and Strain Visualizer__""")
    visualizer()
    
    
    
def pythonpackages():
    st.markdown(r"""# __Useful Python Packages__""")
    text_dict = getText_prep(filename = textfile_path+'pythonpackages.md', split_level = 2)
    
    key='__Introduction__'
    with st.expander(key, expanded=True):
        st.markdown(text_dict[key])   

    key='__Shapely__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])  
        
    key='__SymPy__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])  

    key='__Plotly__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])  

    key='__Rasterio__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])  

    key='__Fenics__'
    with st.expander(key, expanded=False):
        st.markdown(text_dict[key])  
        
    st.write(signatur)
    
#Navigator
if __name__ == '__main__':
    functions = [frontpage, contapprox, tensorfundamentals, elasticity, finiteelementmodelling, fluidfundamental, fluidsinmotion, stressandstrainvisualizer, pythonpackages]
    names = ['1. Frontpage', '2. The Continuum Approximation', '3. Tensor Fundamentals', '4. Elasticity', '5. Finite Element Modelling', '6. Fluids at Rest','7. Fluids in Motion', 'Stress and Strain Visualizer', 'Useful Python Packages']
    navigator(functions, names)