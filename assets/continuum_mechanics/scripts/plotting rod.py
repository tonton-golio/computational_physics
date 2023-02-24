# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:12:33 2023

@author: micha
"""

import numpy as np;
import matplotlib.pyplot as plt;
def fractionabove(p1):
    return 1 - (0.024*11 + 0.976*p1)



p0 = 1
p1 = 0.65
p2 = 11
print((0.9*p0 - p2)/(p1-p2))

print(0.024*p2 + 0.976*p1)

p = np.linspace(0,1,1000)

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(p,fractionabove(p),color='green')
ax.set_xlabel('Density')
ax.set_ylabel('Fraction above water')
ax.set_title('Fraction of rod above water as a function of the density of wood used.')
ax.axhline(y=0,color='blue',label='Sinking line')
ax.legend(loc='best')

plt.show()