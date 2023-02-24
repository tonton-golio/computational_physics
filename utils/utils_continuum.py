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
    
    stress = np.array([
        [0,0,0.5],
        [0,0,0.5],
        [0.5,0.5,-1]
        ])

    young = 1
    poisson = 0.5
    resolution = 3
    scaling = 3
    X, Y, Z = np.linspace(-scaling,scaling,resolution), np.linspace(-scaling,scaling,resolution), np.linspace(-scaling,scaling,resolution)
    strain = straintensor(stress,young,poisson)
    
    eigvalue, eigbasis = np.linalg.eig(strain)
    eigvectors = eigbasis.copy()
    for i in range(3):
        eigvectors[:,i] *= eigvalue[i]
    newcorners = corners @ eigvectors
    
    st.write(f'The stress tensor is {stress}')
    st.write(f'with eigenbasis {eigbasis}, and eigenvalues {eigvalue}')
    
    fig = go.Figure(data=go.Scatter3d(x=corners[:,0],y=corners[:,1],z=corners[:,2],mode='markers'))
    fig.add_trace(go.Scatter3d(x=newcorners[:,0],y=newcorners[:,1],z=newcorners[:,2],mode='markers'))
    
    for i in X:
        for j in Y:
            for k in Z:
                norm = np.linalg.norm(i*eigvalue[0]*eigbasis[:,0] + j*eigvalue[1]*eigbasis[:,1] + k*eigvalue[2]*eigbasis[:,2])
                #st.write(norm)
                u, v, w = (i*eigvalue[0]*eigbasis[:,0] + j*eigvalue[1]*eigbasis[:,1] + k*eigvalue[2]*eigbasis[:,2])/norm
                fig.add_trace(go.Cone(x=np.array(i),y=np.array(j),z=np.array(k), u=np.array(i*eigvalue[0]), v=np.array(j*eigvalue[1]), w=np.array(k*eigvalue[2])))
    st.plotly_chart(fig, use_container_width=True)
    #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    