import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


'# Online and Reinforcement Learning'

text_intro = """
*Allegedly, this is a hard course... So I'll prep a little*


"""


# Hoeffding's inequality
r"""
#### Hoeffding's inequality
Let $X_1, \cdots, X_n$ be independent random variables such that $a_{i}\leq X_{i}\leq b_{i}$ almost surely. Consider the sum of these random variables,

$$
    S_n = X_1 + \cdots + X_n.
$$
Then Hoeffding's theorem states that, for all t > 0,

"""
#$$
#\operatorname{P}
# \left( S_{n}-\mathrm {E} \left[S_{n}\right]\geq t\right) 
#\leq 2\exp \left(-{\frac {2t^{2}}{\sum _{i=1}^{n}(b_{i}-a_{i})^#{2}}}\right)
#$$
r"""
$$
\operatorname {P} \left(\left|S_{n}-\mathrm {E} \left[S_{n}\right]\right|\geq t\right)\leq 2\exp \left(-{\frac {2t^{2}}{\sum _{i=1}^{n}(b_{i}-a_{i})^{2}}}\right)
$$
"""

n_exp = 10
n = 100
Xs = []
ab = np.random.randn(n,2)*2+10
ab = np.sort(ab, axis=1)
for i in range(n_exp):
    Xs.append(np.random.uniform(ab[:,0],ab[:,1]))
Xs = np.array(Xs)

'Xs.shape', Xs.shape
def plot():
    fig = plt.figure(figsize=(6,3))
    plt.hist(ab[:,0], alpha=.6)
    plt.hist(ab[:,1], alpha=.6)

    plt.close()
    st.pyplot(fig)


S = np.sum(Xs, axis=1)
'S.shape', S.shape
'ab.shape',ab.shape

E = np.mean(ab, axis=1)
"E.shape", E.shape
delta = abs(S-E)
t = 20
LHS = sum(delta >= t) / n
f'LHS = {LHS}'

RHS = 2*np.exp(-2*t**2 / sum((ab[:,1]-ab[:,0])**2))
f'RHS = {RHS}'


