import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# we will represent the iceberg as a polygon. So we will import shapely as it has some useful tools for working with polygons.  
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import clip_by_rect
#we want to use a standard ODE solver to evolve in time.
from scipy.integrate import odeint
from time import sleep

gconst = -9.82
rho_ice = 900
rho_ocean = 1030 

def moment_of_inertia(poly,rho=rho_ice):
    x,y = poly.exterior.coords.xy
    x = np.array([*x,x[0]])
    y = np.array([*y,y[0]])
    # See https://en.wikipedia.org/wiki/Second_moment_of_area#Any_polygon
    Iy = np.sum((x[:-1]*y[1:] - x[1:]*y[:-1]) * (x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2))/12
    Ix = np.sum((x[:-1]*y[1:] - x[1:]*y[:-1]) * (y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2))/12
    # Use Perpendicular axis theorem: 
    J = np.abs(Ix + Iy)*rho
    return J

def dStatedt(state,t=0):
    """
    this function returns dState/dt given a State-vector for the iceberg. 
    
    It only takes the time as an input because then it is compatible with odeint which you can use below. 
    """
    x, y, angle, vx, vy, vangle = state
    #move the iceberg into position described by state:
    iceberg = affinity.rotate(iceberg0, angle)
    iceberg = affinity.translate(iceberg, yoff = y, xoff = x)
    M_body = iceberg.area * rho_ice
    
    #Calculate forces
    waterberg = clip_by_rect(iceberg, -20, -20, 20, 0)
    watermass = rho_ocean * waterberg.area
    Fb = - gconst * watermass   #[N]
    Fg = gconst * M_body        #[N]
    
    Fy = Fg + Fb
    
    Fx = 0
    vx = 0
    x = 0
    #dy is the vector between g and b points.
    b = waterberg.centroid
    g = iceberg.centroid
    dy = b.y - g.y
    
    M = dy * Fy
    J_berg = moment_of_inertia(iceberg)
    
    return np.array([vx, vy, vangle, Fx/M_body, Fy/M_body, M/J_berg])

iceberg0 = Polygon([
    [5,12],
    [18,18],
    [13,14],
    [11,6],
    [4,6],
    #MAKE YOUR OWN POLYGON. The units should are metres.
])
#lets shift the polygon so that its centroid is in 0,0:
#Note: centroid is the same as center of mass if the object has constant density.
c = iceberg0.centroid
iceberg0 = affinity.translate(iceberg0, xoff=-c.x, yoff=-c.y)

state = np.array([
    0,0, #position
    0, #rotation
    0,0, #velocity
    0 #angular velocity
    ])

def visualize_state(state, ax = None):
    x, y, angle, vx, vy, vangle = state
    #move the iceberg into position described by state:
    iceberg = affinity.rotate(iceberg0, angle)
    iceberg = affinity.translate(iceberg, yoff = y, xoff = x)
    xy = np.matrix(iceberg.exterior.coords.xy).T
    if ax is None:
         ax = plt.gca()
         ax.cla()
         ax.set_ylim(-20,20)
         ax.set_xlim(-20,20)
         ax.axhline(y=0)
         ax.fill(xy[:,0],xy[:,1], color='grey')
         ax.plot(x,y,'x',color='black')
    else:
         ax.cla()
         ax.set_ylim(-20,20)
         ax.set_xlim(-20,20)
         ax.axhline(y=0)
         ax.fill(xy[:,0],xy[:,1], color='grey')
         ax.plot(x,y,'x',color='black')
    
#test the function on the initial state vector.

fig, ax1 = plt.subplots(figsize=(5,5))

print('solving...')
t = np.linspace(0, 100, 100)
sol = odeint(dStatedt, state, t)
print(sol[10,:])
visualize_state(sol[10,:],ax1)
plt.show()

"""
print('animate')
fig = plt.figure()   
for i in range(len(sol)): 
    visualize_state(sol[i,:])
    sleep(0.1)
    fig.canvas.draw()
"""