import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


'# Online and Reinforcement Learning'

text_intro = """
*Pre-start preperations*
"""

st.markdown(text_intro)

tabs = st.tabs(["Hoeffding's inequality",
                "Markov’s inequality",
                "Chebyshev's inequality",
                "Illustration"])

with tabs[0]: # Hoeffding's inequality
    r"""
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
    cols = st.columns(3)


    n_exp = cols[0].slider('n_exp', 1, 100, 1)
    n = cols[1].slider('n',10, 1000, 100, 10)
    t = cols[2].slider('t', 0.0, 10., 1.,)
    Xs = []
    ab = np.random.randn(n,2)*2+10
    ab = np.sort(ab, axis=1)
    
    for i in range(n_exp):
        Xs.append(np.random.uniform(ab[:,0],ab[:,1]))
    Xs = np.array(Xs)

    def plot():
        fig = plt.figure(figsize=(6,3))
        plt.hist(ab[:,0], alpha=.6)
        plt.hist(ab[:,1], alpha=.6)

        plt.close()
        st.pyplot(fig)


    S = np.sum(Xs, axis=1)

    
    E = np.sum(np.mean(ab, axis=1))
    delta = abs(S-E)
    
    LHS = sum(delta >= t) / n_exp
    

    RHS = 2*np.exp(-2*t**2 / sum((ab[:,1]-ab[:,0])**2))
    f"""
    $$
    {LHS} \leq {RHS}
    $$
    """
    

with tabs[1]: #Markov’s inequality
    r"""
    Markov's inequality gives an upper bound for the probability that a non-negative function of a random variable is greater than or equal to some positive constant.[wikipedia]

    If $X$ is a nonnegative random variable and $a > 0$, then the probability that $X$ is at least $a$ is at most the expectation of $X$ divided by $a$:
    $$
    \operatorname {P} (X\geq a)\leq {\frac {\operatorname {E} (X)}{a}}.
    $$
    """


with tabs[2]: # Chebyshev's inequality
    r"""
    Only a definite fraction of values will be found within a  specific distance from the mean of a distribution. 
    $$
    \Pr(|X-\mu |\geq k\sigma )\leq {\frac {1}{k^{2}}}
    $$
    """

with tabs[3]:
    'illu'
    
