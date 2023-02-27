from utils.utils_global import *
import plotly.graph_objects as go
from time import sleep

textfile_path = 'assets/continuum_mechanics/text/'
imagefile_path = 'assets/continuum_mechanics/images/'

signatur = """*If you wish to contribute, report errors, have feedback or comments, kindly contact*
    
    Michael Bach - Discord: bach#2630 - lvp115@alumni.ku.dk"""

def straintensor(stress,young,poisson):
    """Takes inputs:
       stress: N by N np.array 2D matrix.
       young: Float corresponding to the Young's modulus
       poisson: float corresponding to the poisson ratio
       Returns the corresponding strain tensor.
       """
    return stress * (1 + poisson) / young - np.identity(len(stress)) * stress.trace() * (poisson/young)

def visualizer():
    #Coordinate corners of 2 by 2 square, each row is a coordinate 
    corners = np.array([
        [1,-1,-1],
        [-1,-1,-1],
        [1,1,-1],
        [-1,1,-1],
        [1,-1,1],
        [-1,-1,1],
        [1,1,1],
        [-1,1,1], 
        ])
    
    #Plotting parameters, scaling defines range to plot as [-scaling:scaling]
    #Resolution defines the amount of cones drawn along each axis. 
    resolution = 3
    scaling = 3
    X, Y, Z = np.linspace(-scaling,scaling,resolution), np.linspace(-scaling,scaling,resolution), np.linspace(-scaling,scaling,resolution)
    
    #Stress tensor input
    st.write('**Stress tensor input**')
    inputs = st.columns([1,1,1,0.1,2])
    s11 = inputs[0].number_input(r'$\sigma_{xx}$', value = 0)
    s12 = inputs[0].number_input(r'$\sigma_{yx}$', value = 0)
    s13 = inputs[0].number_input(r'$\sigma_{zx}$', value = 0)
    s21 = inputs[1].number_input(r'$\sigma_{xy}$', value = 0)
    s22 = inputs[1].number_input(r'$\sigma_{yy}$', value = 0)
    s23 = inputs[1].number_input(r'$\sigma_{zy}$', value = 0)
    s31 = inputs[2].number_input(r'$\sigma_{xz}$', value = 0)
    s32 = inputs[2].number_input(r'$\sigma_{yz}$', value = 0)
    s33 = inputs[2].number_input(r'$\sigma_{zz}$', value = 1)
    
    #Collect in stress tensor
    stress = np.array([
        [s11,s21,s31],
        [s12,s22,s32],
        [s13,s23,s33]
        ])
    
    inputs[4].write('**Material properties**')
    young = inputs[4].number_input('Young\'s Modulus', min_value = 0, value = 1)
    poisson = inputs[4].number_input('Poisson\'s Ratio', min_value = float(-1), max_value = float(0.5), value = float(0.5))
    
    #Compute strain tensor and solve the eigenproblem.
    strain = straintensor(stress,young,poisson)
    eigvalue, eigbasis = np.linalg.eig(strain)
    eigvectors = eigbasis.copy()
    
    #Compute the corner coordinates of deformed box - NEEDS CORRECTION!
    for i in range(3):
        eigvectors[:,i] *= eigvalue[i]
    newcorners = corners @ eigvectors
    
    #Plot deformed and undeformed box corners
    fig = go.Figure(data=go.Scatter3d(x=corners[:,0],y=corners[:,1],z=corners[:,2],mode='markers'))
    fig.add_trace(go.Scatter3d(x=newcorners[:,0],y=newcorners[:,1],z=newcorners[:,2],mode='markers'))
    
    #Plot cones.
    for i in X:
        for j in Y:
            for k in Z:
                norm = np.linalg.norm(i*eigvalue[0]*eigbasis[:,0] + j*eigvalue[1]*eigbasis[:,1] + k*eigvalue[2]*eigbasis[:,2])
                #st.write(norm)
                u, v, w = (i*eigvalue[0]*eigbasis[:,0] + j*eigvalue[1]*eigbasis[:,1] + k*eigvalue[2]*eigbasis[:,2])/norm
                fig.add_trace(go.Cone(x=np.array(i),y=np.array(j),z=np.array(k), u=np.array(i*eigvalue[0]), v=np.array(j*eigvalue[1]), w=np.array(k*eigvalue[2])))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f'Eigenbasis of the stress tensor is {eigbasis} with eigenvalues {eigvalue}. Remember that Youngs Modulus and the stresses must have the same units. **Remember to add description of figure**')
    #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

def femvis():
    st.write('util part')
    
    
    
    
    
    
    
    