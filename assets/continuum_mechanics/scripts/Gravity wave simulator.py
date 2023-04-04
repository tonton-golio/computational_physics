import numpy as np;

lam = 1 #[m]
h = 0.5 #[m]
d = 10 #[m]
g = 9.81 #[accel]
k = 2 * 3.1415 / lam

cshallow = np.sqrt( g * d)
print(f'Shallow water speed is {cshallow}')

cphasefinite = np.sqrt(g * np.tanh( k*d ) / k)
cgroupfinite = cphasefinite * (1 + 2*k*d/np.sinh(2*k*d)) / 2
print(f'Finite water speed is {cgroupfinite}')

dispdeep = np.sqrt( g * k)
cphasedeep = dispdeep / k
cgroupdeep = cphasedeep / 2
print(f'Deep water speed is {cgroupdeep}')

print(f'Time to make distance 200m = {200/cgroupfinite} [s]')