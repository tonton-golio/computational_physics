from utils.utils_global import *


def landingPage():
    ''
    """# Continuum Mechanics"""
    #text_dict = getText_prep(filename = text_path+'landingPage.md', split_level = 1)
    #for key in text_dict:
    #    text_dict[key]

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
    1. Stress and Strain
    2. Elasticity
    3. Finite element modelling
    4. Continuum Dynamics and Ideal flows
    5. Viscosity and Gravity waves
    6. Pipe flow
    7. Navier Stokes Equations
    '''
    
# Navigator
topic_dict = {"Landing Page" : landingPage}

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()
