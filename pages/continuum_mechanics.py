from utils.utils_global import *


set_rcParams()

def frontpage():
    ''
    st.title("""Continuum Mechanics""")
    #text_dict = getText_prep(filename = text_path+'landingPage.md', split_level = 1)
    #for key in text_dict:
    #    text_dict[key]

    st.write("""
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
    """)

    cols = st.columns(3)
    cols[0].image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FVortex&psig=AOvVaw1nOAFgaJ6XOZyXfZkmfmac&ust=1675880067257000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOiA4_KBhP0CFQAAAAAdAAAAABAR')
    cols[0].caption('Vortex around an airplane')

    cols[1].image('https://i.pinimg.com/originals/49/f3/94/49f3946a610b9a3665c9e2c9bd4c571c.jpg')
    cols[1].caption('')

    cols[2].image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Amadeo_Avogadro.png/248px-Amadeo_Avogadro.png')
    cols[2].caption('Amedeo Avogadro')

    st.title("""Table of contents""")
    st.write("""
    1. The Continuum Approximation
    2. Elasticity
    3. Finite element modelling
    4. Continuum Dynamics and Ideal flows
    5. Viscosity and Gravity waves
    6. Pipe flow
    7. Navier Stokes Equation
    """)

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
    st.title('The Continuum Approximation')
    st.write(r"""
    The physics of continuum mechanics, requires that a mass of particles can be 
    modelled as continuous matter where the particles are infinitesimal. 
    
    $\textit{
    "Whether a given number of molecules is large enough to warrant the use of 
    a smooth continuum description of matter depends on the desired precision. 
    Since matter is never continuous at sufficiently high precision, continuum 
    phyiscs is always an approximation. But as long as the fluctuations in 
    physical quantities caused by the discreteness of matter are smaller than 
    the desired precision, matter may be taken to be continuous. To observe the 
    continuity, one must so to speak avoid looking too sharply at material bodies. 
    Fontenelle stated in a similar context that 
    "Science originates from curiosity and bad eyesight"."
    }$
    
    From Physics of Continuous Matter 2nd edition by Benny Lautrup
    
    """)
    
    st.title('Gases')
    st.write(r"""
    
    Density fluctuations
    
    The density of a pure gas is given as
    $$
    \rho = \frac{N m}{V}
    $$
    Where $N$ is the number of molecules, $V$ is the volume and $m$ is the mass of the molecules
    
    From general statistics, it can be shown that the fluctuations in $N$ follows 
    the RMS of the number of molecules and since the density is linear depended 
    on $N$ this gives;
    
    $$
    \frac{\Delta \rho}{\rho} = \frac{\Delta N}{N} = \frac{1}{\sqrt{N}}
    $$
    
    So if we require a relative precision of $\epsilon = 10^{-3} > \frac{\Delta \rho}{\rho} $ in the density fluctuations 
    there must be $N > \epsilon^^{-2}$ molecules. They occupy a volume of $\epsilon^{-2}L_{molecule}^3$, 
    where $L_{molecule}$ is the molecular seperation length. 
    At the scale of $L_{mol}$ the continuum approximation completely breaks down, 
    and to ensure correct approximation within a certain precision the minimum cell size 
    that we consider is given as;
    
    $$
    L_{micro} = $\epsilon^{-2}L_{molecule}^3$
    $$
    $$
    L_{mol} =\left( \frac{V}{N} \right)^{1/3} = \left( \frac{M_{molecule}}{\rho N_A} \right)^{1/3} 
    $$
    Where $L_{micro}$ is the sidelength of the cubic cell that satisfies the 
    precision condition, $M_{molecule}$ is the molar mass of the substance.
    """)
    st.image('computational_physics\assets\continuum_mechanics\images\densityfluctuations.png', 
            width=300, caption='From Physics of Continuous Matter 2nd edition by Benny Lautrup')
    
    st.write(r"""
    Macroscopic Smoothness
    Another criteria for the continuum approximation is the Macroscopic Smoothness. 
    We require the relative change in density between cells to be less than the 
    precision $\epsilon$ along any direction.
    
    $$
    \left( \frac{\partial \rho}{\partial x} \right) < \frac{\rho}{L_{macro}}
    $$
    where $L_{macro} = \epsilon^{-1} L_{micro}$
    
    if the above is fulfilled, the change in density can be assumed to vary smooth, 
    and the continuum approximation holds. However, the thickness of interfaces between 
    macroscopic bodies are typically on the order of $L_{molecule}} and not $L_{macro}, 
    we instead represent these as surface discontinuities.
             
         
             """)
    
    st.write(r"""
    Velocity fluctuations
    
    
             """)
    st.write(r"""
    Mean-free-path
    
             """)
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
    'The Continuum Approximation' : contapprox,
    'Testpage'  : testpage,
              }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()
