from utils.utils_inverse import *
import streamlit_toggle as tog
from scipy.special import rel_entr
import string

set_rcParams()

def frontpage():
    ''
    """# Continuum Mechanics"""
    text_dict = getText_prep(filename = text_path+'landingPage.md', split_level = 1)
    for key in text_dict:
        text_dict[key]

    """
    In the macroscopic world, most materials that surround us e.g. solids and 
    liquids can safely be assumed to exist as continua, that is, the materials 
    completely fill the space they occupy and the underlying atomic structures 
    can be neglected. This course offers a modern introduction to the physics of
    continuous matter with an emphasis on examples from natural occurring 
    systems (e.g. in the physics of complex systems and the Earth sciences). 
    Focus is equally on the underlying formalism of continuum mechanics and 
    phenomenology. In the course you will become familiar with the mechanical 
    behavior of materials ranging from viscous fluids to elastic solids. 

    A description of the deformation of solids is given, including the concepts
    of mechanical equilibrium, the stress and strain tensors, linear elasticity.
    A derivation is given of the Navier-Cauchy equation as well as examples 
    from elastostatics and elastodynamics including elastic waves. A 
    description of fluids in motion is given including the Euler equations, 
    potential flow, Stokes' flow and the Navier-Stokes equation. Examples of 
    numerical modeling of continous matter will be given.
    
    From: https://kurser.ku.dk/course/nfyk10005u
    """

    cols = st.columns(3)
    cols[0].image('https://nci-media.cancer.gov/pdq/media/images/428431-571.jpg')
    cols[0].caption('magnetic resonance imaging (MRI)')

    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png')
    cols[1].caption('Amedeo Avogadro.')


    cols[2].image('https://i.pinimg.com/originals/49/f3/94/49f3946a610b9a3665c9e2c9bd4c571c.jpg')
    cols[2].caption('')

    '''
    Navigate between the follwoing topics in the left menu.
    1. Criteria for Continua
    2. Elasticity
    3. Finite element modelling
    4. Continuum Dynamics and Ideal flows
    5. Viscosity and Gravity waves
    6. Pipe flow
    7. Navier Stokes Equation
    '''

def testpage():
    ''
    cols = st.columns(2)
    cols[0].title('Test title')
    
    # intro 
    cols[0].write(r"""
    Sample text 1
    """)
    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png', 
            width=300, caption='Avada Avogadro')

    '---'

    st.write(r"""
    Sample text 2
    $$
         L(a)T^e_x = \int_0^{\infty},
    $$
    Sample text 3
    """)
    st.caption('Sample caption')

def contapprox():
    ''
    cols = st.columns(2)
    cols[0].title('The Continuum Approximation')
    
    # intro 
    cols[0].write(r"""
    The physics of continuum mechanics, requires that a mass of particles can be 
    modelled as continuous matter where the particles are infinitesimal. 
    
    \textit{
    "Whether a given number of molecules is large enough to warrant the use of 
    a smooth continuum description of matter depends on the desired precision. 
    Since matter is never continuous at sufficiently high precision, continuum 
    phyiscs is always an approximation. But as long as the fluctuations in 
    physical quantities caused by the discreteness of matter are smaller than 
    the desired precision, matter may be taken to be continuous. To observe the 
    continuity, one must so to speak avoid looking too sharply at material bodies. 
    Fontenelle stated in a similar context that 
    "Science originates from curiosity and bad eyesight"."
    }
    
    From Physics of Continuous Matter 2nd edition by Benny Lautrup
    
    """)
    cols[1].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png', 
            width=300, caption='Avada Avogadro')
    st.write(r"""
    The Continuum Approximation for Gases
    
    Density fluctuations
    
    Macroscopic smoothness
    
    Velocity fluctuations
    
    Mean free path
    
    """)
    '---'

    st.write(r"""
    Sample text 2
    $$
         L(a)T^e_x = \int_0^{\infty},
    $$
    Sample text 3
    """)
    st.caption('Sample caption')


# Navigator
topic_dict = {
    "Frontpage" : frontpage,
    'Testpage'  : testpage,
    'The Continuum Approximation' : contapprox,
              }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()