import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
try: import graphviz # having trouble with this when hosted
except: pass
try: import networkx as nx  # networkx too :(
except: pass
import time
import sys
sys.setrecursionlimit(15000)

st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)

# setting matplotlib style:

mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['axes.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'
# mpl.rcParams['axes.grid'] = True  # should we?


def run_stat_mech():
    # Sidebar
    with st.sidebar:
        size = st.slider('size',3,100,10)
        beta = st.slider('beta',0.01,5.,1.)
        nsteps = st.slider('nsteps',3,10000,100)
        nsnapshots = 4

    # functions
    def metropolisVisualization():
        dEs = np.linspace(-1,3,1000)
        prob_change = np.exp(-beta*dEs)
        prob_change[dEs<0] = 1

        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(dEs, prob_change, color='pink', lw=7)
        ax.set_ylabel('probability of acceptance', color='white')
        ax.set_xlabel('Energy difference', color='white')
        ax.set(xticks=[0])
        plt.grid()

        return fig

    def ising():
        # initialize
        X = np.random.rand(size,size)
        X[X>0.5] =1 ; X[X!=1] =-1
        E = 0 
        for i in range(size):
            for j in range(size):
                sum_neighbors = 0
                for pos in [(i,(j+1)%size),    (i,(j-1+size)%size), 
                               ((i+1)%size,j), ((i-1+size)%size,j)]:
                    
                    sum_neighbors += X[pos]
            E += -X[i,j] * sum_neighbors/2

        results = {"Energy" : [E], 
                   "Magnetization" : [np.sum(X)], 
                   "snapshots": {} }
        for step in range(nsteps):
            (i,j) = tuple(np.random.randint(0,size-1,2)) #choose random site

            sum_neighbors = 0
            for pos in [(i,(j+1)%size),    (i,(j-1+size)%size), 
                           ((i+1)%size,j), ((i-1+size)%size,j)]:
                sum_neighbors += X[pos]
                
            dE = 2 *X[i,j] * sum_neighbors

            
            if np.random.rand()<np.exp(-beta*dE):
                X[i,j]*=-1
                E += dE

            results['Energy'].append(E.copy())
            results['Magnetization'].append(np.sum(X))
            if step in np.arange(nsnapshots)*nsteps//nsnapshots:
                results['snapshots'][step]=X.copy()

        # load, fill and save susceptibility data
        try: data = np.load('pages/data.npz', allow_pickle=True)[np.load('pages/data.npz', allow_pickle=True).files[0]].item()
        except: data = {};  np.savez('pages/data', data)

        susceptibility = np.var(results['Magnetization'][-nsteps//4*3:])
        data[beta] = {'sus': susceptibility, 'nsteps':nsteps, 'size':size}
        np.savez('pages/data', data)
        return results, data

    def plotSnapshots(nsnapshots = 4):
        fig, ax = plt.subplots(1,nsnapshots, figsize=(15,3))
        for idx, key in enumerate(results['snapshots'].keys()):
            ax[idx].imshow(results['snapshots'][key])
        return fig

    def plotEnergy_magnetization():
        fig, ax = plt.subplots(2,1, figsize=(5,6))
        ax[0].plot(results['Energy'],c='purple')
        ax[1].plot(results['Magnetization'], color='orange')
        

        for i in [0,1]:
            ax[i].set_xlabel('Timestep', color='white')
        
        ax[0].set_ylabel('Energy', color='white')
        ax[1].set_ylabel('Magnetization', color='white')
        
        return fig
    
    def plotSusceptibility():
        ## susceptibility plot

        fig, ax = plt.subplots( figsize=(5,3))
        ax.scatter(x = list(data.keys()), 
                      y = [data[key]['sus'] for key in data.keys()],
                      s = [data[key]['size'] for key in data.keys()],
                      color='cyan')
        ax.set_ylabel('Susceptibility', color='white')
        ax.set_xlabel('beta', color='white')

        return fig

    # Render

    st.markdown(r"""
        # Statistical Mechanics
        
        ## Partition function
        
        ### Microcanonical Ensemble
        The central assumption of statistical mechanics is "principle of 
        equal a priori probabilities" which argues that all (quantum) states 
        with same energy $E$ of the closed mactoscopic system exists equally 
        likely.
        With this assumption one can say that the system become particular state
        $i$ with probability
        $$
            P_i = \frac{1}{\Omega}
        $$
        Here $\Omega$ is the total number of (quantum) microstate of the 
        system with energy $E$.


        According to Ludwig Boltzmann, entropy of system with microcanonical 
        ensemble of the system with fixed energy $E$ is expressed as
        $$
            S = k_\mathrm{B} \ln \Omega.
        $$ 
        Here $k_\mathrm{B}$ is Boltzmann's constant.
        
        In this system, temperature $T$ is statistically defined as 
        $$ 
        \begin{align*}
            \frac{1}{T} 
            &= \frac{\partial S}{\partial E} 
            \\&= k_\mathrm{B} \frac{\partial}{\partial E} \ln \Omega
        \end{align*}
        $$ 

        ### Canonical Ensemble
        #### Temperature of system and reservoir become same in equilibrium
        Let's consider subsystem of whole system of microcanonical ensemble. 
        For simplicity, we only consider only one subsystem and assuming it 
        is small enough comparing to the rest of the system. Let's say this 
        small part as 
        "system" and rest of enoumous part of the original microcanonical 
        system as "reservoir" or "heat bath".
        By defining the system's energy as $E_\mathrm{s}$ and reservoir's 
        energy as $E_\mathrm{r}$, we consider the energy exchange 
        between these two. Because total energy conserved, the sum of two 
        energy is constant value.
        $$ 
            E_\mathrm{t} = E_\mathrm{s} + E_\mathrm{r} 
        $$
        Here $E_\mathrm{t}$ is total energy of whole system.

        Let's think about number of state.
        $\Omega(E_\mathrm{t})$ is the total number of states with energy 
        $E_\mathrm{t}$.
        $\Omega(E_\mathrm{s}, E_\mathrm{r})$ is the total number of states with 
        system has energy $E_\mathrm{s}$ and system has energy $E_\mathrm{r}$.
        It can be the product of total number of state of system and reservoir.
        $$
            \Omega(E_\mathrm{s}, E_\mathrm{r})
            = \Omega_\mathrm{s}(E_\mathrm{s}) \Omega_\mathrm{r}(E_\mathrm{r})
        $$
        Entropy of whole system become
        $$
            S_\mathrm{t} =  k_\mathrm{B} \ln \Omega(E_\mathrm{s}, E_\mathrm{r}).
        $$
        Entropy of the system and reservoir become
        $$
            S_\mathrm{s} =  k_\mathrm{B} \ln \Omega(E_\mathrm{s}),
        $$
        $$
            S_\mathrm{r} =  k_\mathrm{B} \ln \Omega(E_\mathrm{r}).
        $$
        Thus total entropy become sum of system and reservoir.
        $$
            S_\mathrm{t} = S_\mathrm{s} + S_\mathrm{r}.
        $$

        Probability of finding the state system energy $E_\mathrm{s}$ and 
        reservoir energy $E_\mathrm{r}$ is
        $$
        \begin{align*}
            P(E_\mathrm{s}, E_\mathrm{r}) 
            &= \frac{\Omega(E_\mathrm{s}, E_\mathrm{r})}{\Omega(E_\mathrm{t})}
            \\&= \frac{\Omega_\mathrm{s}(E_\mathrm{s}) 
            \Omega_\mathrm{r}(E_\mathrm{r})}
            {\Omega(E_\mathrm{t})}
        \end{align*}
        $$
        Most probable state (thermodynamical equilibrium state) satisfies
        $$
            \frac{\partial P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
            =0
        $$
        This condition does not change with logarithmic scale.
        $$
            \frac{\partial \ln 
            P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
            =0
        $$
        Using number of state, this condition can be expressed as 
        $$
        \begin{align*}
            0 &= 
                \frac{\partial \ln 
                P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
              \\&= 
                \frac{\partial}{\partial E_\mathrm{s}} 
                \ln \frac{\Omega_\mathrm{s}(E_\mathrm{s}) 
                \Omega_\mathrm{r}(E_\mathrm{r})}{\Omega(E)}
              \\&= 
                \frac{\partial}{\partial E_\mathrm{s}} 
                \ln \Omega_\mathrm{s}(E_\mathrm{s}) 
                +
                \frac{\partial}{\partial E_\mathrm{r}} 
                \frac{\partial E_\mathrm{r}}{\partial E_\mathrm{s}} 
                \ln \Omega_\mathrm{r}(E_\mathrm{r})
              \\&= 
                \frac{1}{k_\mathrm{B}}
                \frac{\partial S_\mathrm{s}}{\partial E_\mathrm{s}} 
                -
                \frac{1}{k_\mathrm{B}}
                \frac{\partial S_\mathrm{r}}{\partial E_\mathrm{r}} 
              \\&= 
                \frac{T_\mathrm{s}}{k_\mathrm{B}}
                -
                \frac{T_\mathrm{r}}{k_\mathrm{B}}.
        \end{align*}
        $$
        Here $T_\mathrm{s}$ is temperature of system and $T_\mathrm{r}$ is 
        temperature of reservoir.
        From this result, we can say in equilibrium system's temperature 
        and reservoir's temperature is same.
        $$
                T_\mathrm{s} = T_\mathrm{r}

        $$
        If two system and reservoir exchange the energy, they have same 
        temperature.

        #### Boltzmann distribution 
        When system is in state $i$ and has energy $E_i$, reservoir has energy
        $E_\mathrm{t} - E_i$, probability of happening this state $i$ is 
        $$
        \begin{align*}
            P_i 
            &\propto 
                \Omega_\mathrm{r}(E_\mathrm{t} - E_i) 
            \\&= 
                \exp 
                \left[ \frac{1}{k_\mathrm{B}} S_r(E_\mathrm{t} - E_i) \right]
            \\&\approx 
                \exp \left[ \frac{1}{k_\mathrm{B}} 
                \left(S_\mathrm{r}(E_\mathrm{t}) 
                - \left. 
                \frac{\mathrm{d}S_\mathrm{r}}{\mathrm{d}E} 
                \right|_{E=E_\mathrm{t}}E_i\right)
                \right]
            \\&= 
                \exp \left[ \frac{1}{k_\mathrm{B}} 
                \left(S_\mathrm{r}(E_\mathrm{t}) 
                - \frac{E_i}{T_\mathrm{r}}\right)
                \right]
            \\&\propto 
                \exp \left( 
                -\frac{E_i}{k_\mathrm{B}T_\mathrm{r}} 
                \right)
        \end{align*}
        $$
        We used Taylor series expansion because $E_\mathrm{t}\gg E_i$.
        The term 
        $\exp \left(-\frac{E_i}{k_\mathrm{B}T} \right)$
        called "Boltzmann distribution".
        The probability that state $i$ happens is proportional to Boltzmann 
        distribution
        $P_i \propto \exp \left(-\frac{E_i}{k_\mathrm{B}T} \right)$.

        #### Partition function, free energy and thermodynamical observable
        The partition function is defined as the sum of the Boltzmann factor 
        of all states of system.
        $$
        \begin{align*}
            Z &= \sum_i \exp \left({-\frac{E_i}{k_\mathrm{B}T}} \right)
              \\&= \sum_i \exp \left({-\beta E_i}\right).
        \end{align*}
        $$
        Here $\beta = \frac{1}{k_\mathrm{B}T}$.
        Thus probability of finding system with state $i$ with energy $E_i$ 
        becomes
        $$
        \begin{align*}
            P_i 
            &=
            \frac
            {\exp \left(-\frac{E_i}{k_\mathrm{B}T} \right)}
            {Z} 
            \\&=
            \frac
            {\exp \left(-\beta E_i \right)}
            {Z} .
        \end{align*}
        $$

        Using the partiton the functions, we are able obtain any thermodynamical
        observable. 
        A particularly important value we may obtain is the (Helmholtz) free 
        energy,
        $$
        \begin{align*}
            F 
            &= -k_\mathrm{B}T \ln Z 
            \\&= -\frac{1}{\beta}\ln Z 
            \\&= \left<E\right> - TS.
        \end{align*}
        $$
        We can also obtain average energy from partition function.
        $$
        \begin{align*}
            \left<E\right>
            &=
            \sum_i E_i P_i
            \\&=
            \frac{1}{Z} \sum_i E_i \exp \left({-\beta E_i}\right)
            \\&=
            -\frac{1}{Z} \sum_i \frac{\partial}{\partial \beta} 
            \exp \left({-\beta E_i}\right)
            \\&=
            -\frac{1}{Z} \frac{\partial}{\partial \beta} Z
            \\&=
            - \frac{\partial}{\partial \beta} \ln Z
        \end{align*}
        $$
        From average energy, we can obtain specific heat.
        $$
        \begin{align*}
            C 
            &=
            \frac{\partial \left<E\right>}{\partial T}
            \\&= 
            \frac{\partial \left<E\right>}{\partial \beta}
            \frac{\partial \beta}{\partial T}
            \\&=
            -\frac{1}{k_\mathrm{B} T^2}
            \frac{\partial \left<E\right>}{\partial \beta}
            \\&=
            \frac{1}{k_\mathrm{B} T^2}
            \frac{\partial^2}{\partial \beta^2} \ln Z
        \end{align*}
        $$
        Specific heat is equal to variace of energy.
        $$
        \begin{align*}
            k_\mathrm{B} T^2C 
            &=
            \frac{\partial^2}{\partial \beta^2} \ln Z
            \\&=
            \frac{\partial}{\partial \beta} 
            \left(
            \frac{\partial}{\partial \beta}
            \ln Z
            \right)
            \\&=
            \frac{\partial}{\partial \beta} 
            \left(
            \frac{1}{Z}
            \frac{\partial Z}{\partial \beta}
            \right)
            \\&=
            -
            \frac{1}{Z^2}
            \left(
            \frac{\partial Z}{\partial \beta}
            \right)^2
            +
            \frac{1}{Z}
            \frac{\partial^2 Z}{\partial \beta^2}
            \\&=
            - \left<E\right>^2 + \left<E^2\right> 
            \\&=
            \left< \left( E - \left< E \right> \right)^2 \right>
            \\&=
            \left(\Delta E\right)^2
        \end{align*}
        $$
        Assuming energy of the system can be approximated by system size 
        $\left< E \right> \sim N k_\mathrm{B} T$ 
        (you know this is good approximation because the energy of 
        particle in a box is $E=\frac{3}{2} N k_\mathrm{B}T$), 
        specific heat is also 
        approximated as $C \sim N k_\mathrm{B}$. 

        Thus
        $$
            \Delta E \sim N^{1/2}.
        $$
        Variance of energy scales as square root of system size.

        From free energy we can obtain the entropy,         
        $$
            S = - \left. \frac{\partial F}{\partial T} \right|_H.
        $$
        From free energy we can also obtain the magnetization,         
        $$
            M = - \left.\frac{\partial F}{\partial H} \right|_T.
        $$
        Susceptibility becomes
        $$
            \chi_T = - \left.\frac{\partial^2 F}{\partial H^2} \right|_T.
        $$
        Notice that susceptibility is equal to variance of magnetization.
        $$
        \begin{align*}
            k_\mathrm{B} T\chi_T 
            &= 
            -k_\mathrm{B} T
            \frac{\partial^2 F}{\partial H^2} 
            \\&= 
            \frac{1}{\beta^2}
            \frac{\partial}{\partial H} 
            \left(
            \frac{\partial}{\partial H} 
            \ln Z
            \right)
            \\&= 
            \frac{1}{\beta^2}
            \frac{\partial}{\partial H} 
            \left(
            \frac{1}{Z} 
            \frac{\partial Z}{\partial H} 
            \right)
            \\&=
            -
            \frac{1}{\beta^2}
            \frac{1}{Z^2}
            \left(
            \frac{\partial Z}{\partial H}
            \right)^2
            +
            \frac{1}{\beta^2}
            \frac{1}{Z}
            \frac{\partial^2 Z}{\partial H^2}
            \\&=
            - \left<M\right>^2 + \left<M^2\right> 
            \\&=
            \left< \left( M - \left< M \right> \right)^2 \right>
            \\&=
            \left(\Delta M\right)^2
        \end{align*}
        $$


        ## Ising Model 
        Ernst Ising introduced the model of ferromagnetism with a discrete 
        magnetic momemt $s_i$.
        He approximated spin (magnetic moment) can be only one of the two 
        state.
        $$
            s_i = 
            \begin{cases}
            +1 \\
            -1
            \end{cases}
        $$
        $s_i = +1$ represents spin up and $s_i = -1$ represents spin down.

        Hamiltonian of Ising model is
        $$
            \mathcal{H} 
            = 
            -J \sum_{\left<i j\right>} s_i s_j - h \sum_i s_i
        $$
        Here, $\left<i j\right>$ means sum is performed over 
        nearest-neighbor interaction which means 
        $\sum_{\left<i j\right>} 
        = \frac{1}{2} \sum_i \sum_{\left< j \right>}$.
        Index of second summation $\left< j \right>$ is nearest neighbor 
        of spin $i$.
        $J>0$ is coupling strength and $h$ is the external magnetic field.
        Notice that $J<0$ indicates antiferromagnetism.
        In case of $J=0$, there is no interspin interaction.
        When $J$ depends on pair of spin, it is spin glass.

        Magnetization of Ising model becomes
        $$
            M = \sum_{i=1}^N s_i.
        $$
        or 
        $$
            m = \frac{1}{N} \sum_{i=1}^N s_i.
        $$
        depending on the problem.

        Caliculating partition function of 1D-lattice Ising model was 
        performed by E. Ising himself, and that of 2D-lattice Ising 
        model was aesthetically performed by Lars Onsager. 
        However, nobody succeed in caliculating partition function of
        3D-lattice Ising model until now as far as I know. 
        Thus, generally speaking, caliculating partition function of 
        Ising model is one of the hardest problem we human being have.
        Nevertheless, we can use computer to numerically caliculate 
        thermodynamic observable of canonical ensemble.
        
        ## Metropolis algorithm
        ### Monte Carlo method
        Monte Carlo is a subpart of Monaco and it is famous for gambling.
        According to Wikipedia, Nicholas Metropolis suggest the name.

        This method is used for numerically estimate physical observable.
        Statistical mechanical average value of physical observable 
        $\left<A\right>$ is 
        $$
        \begin{align*}
            \left<A\right>
            &=
            \frac{\sum_i A_i \exp \left(-\beta E_i \right) }{Z}
            \\&=
            \frac
            {\sum_i A \exp \left(-\beta E_i \right)}
            {\sum_i \exp \left(-\beta E_i \right)}
            \\&=
            \sum_i A_i P_i
        \end{align*}
        $$
        This can be approximated by 
        $$
            \left<A\right>
            \approx 
            \sum_i A_i \tilde{P}_i
        $$
        Here $\tilde{P}_i$ is sampled probability distribution.
        By approximating statistical mechanical probability distribution 
        with sampled probabiolity distribution, we can get approximated 
        physical observable from Monte Carlo method.
        ### Markov process and master equation
        Important assumption of Metropolis algorithm is Markov process
        which argue that state change is only depends on previous state.
        By using this Markov assumption, one can describe the time 
        evolution of probability distribution of state $i$ as 
        master equation.
        $$ 
            \frac{\mathrm{d}P_i}{\mathrm{d}t}
            =
            \sum_j \left( w_{ij}P_j - w_{ji}P_i \right)
        $$ 
        Here $P_i$ is probability of state $i$ and $w_{ij}$ is transition
        rate from state $j$ to state $i$.
        $w_{ij}P_j$ is flux from state $j$ to state $i$.
        First term of master equation shows the incoming probablity flux 
        from state $j$ to state $i$ and second term shows outgoing 
        probability from state $i$ to state $j$.

        Notice that probability normalization 
        $$ 
            \sum_i P_i = 1
        $$ 
        constrains probability flux to be conserved as a form of master 
        equation.

        In steady state, probability distribution does not depend on time 
        i.e. $\frac{\mathrm{d}P_i}{\mathrm{d}t} = 0$.
        Then
        $$ 
            \sum_j \left( w_{ij}P_j - w_{ji}P_i \right)
            = 0 
        $$ 
        which is equivalent to condition of steady state.
        $$ 
            \sum_j w_{ij}P_j = \sum_j w_{ji}P_i
        $$ 
        This argues that, in steady state, total flux coming to state $i$ 
        equals to total flux going out from state $i$.
        However, this condition can allow system to have irreversible 
        state transition (circular state transition) which violate 
        the concept of thermal equilibrium.
        ### Detailed balance
        By using "detailed balance" as a condition of equilibrium,
        one can avoid such irreversible state transition.
        $$ 
            w_{ij}P_j = w_{ji}P_i
        $$ 
        This argues that flux from state $j$ to state $i$ is equal to 
        that of opposite. 
        This condition does not allow irreversible state transition.
        ### Metropolis-Hastings algorithm 
        What Metropolis algorithm do are two things: trying random state 
        transition then accepting that transition with specific criteria.
        How can we set the criteria for sampling state from canonical 
        ensemble?
        We need to restrict transition rate to sample state from canonical
        ensemble.
        By using detailed balance condition, we can know the form of 
        transition rate and it provide us a criteria.
        Equilibrium probability distribution of canonical ensemble is
        $$
            P_i
            =
            \frac
            {\exp \left(-\beta E_i \right)}
            {Z} .
        $$
        By combining this with detailed balance condition
        $$
            \frac{w_{ij}}{w_{ji}} 
            =
            \frac{P_i}{P_j} 
            =
            \frac
            { \frac{\exp \left(-\beta E_i \right)}{Z} }
            { \frac{\exp \left(-\beta E_j \right)}{Z} }
            =
            \exp \left[-\beta \left(E_i - E_j\right) \right]
            =
            \exp \left(-\beta \Delta E_{ij} \right)
        $$
        Here $\Delta E_{ij}$ is energy difference from state $j$ to state
        $i$.

        """)

    cols = st.columns(2)
    cols[0].markdown(r"""
        As I mentioned earlier, first step of Metropolis algorithm is 
        trying random state transition.
        In second step of the algorithm, first caliculate energy difference.
        If energy difference is negative, accept the trial transition.
        If energy difference is positive, accept with weight 
        $\exp \left[-\beta \Delta E_{ij} \right]$. 
        """)
    
    cols[1].pyplot(metropolisVisualization())

    results, data = ising()
     
    st.markdown(r"""
        Below are snapshots of the output of a simulation of the 2d Ising model
        using the metropolis algorithm.
        """)

    st.pyplot(plotSnapshots(nsnapshots = 4))

    cols = st.columns(2)
    cols[0].markdown(r"""If we track paramters through time,
        we may be able to spot a phase transition (they're a rare breed).
        On the right are plots of the energy and magnetization over time. Below
        is susceptibility as obtained the variance of the magnetization, 
        $\chi = \left< \left< M\right> - M\right>$ (:shrug)""")
    cols[1].pyplot(plotEnergy_magnetization())
    cols[0].pyplot(plotSusceptibility())
    
def run_phaseTransitions_CriticalPhenomena():
    st.markdown(r"""
    # Phase transitions & Critical phenomena
    ## Mean Field Solution to Ising Model
    ### Hamiltonian and partition function
    Hamiltonian and partition function of Ising model are
    $$
        \mathcal{H} 
        = 
        -J \sum_{\left<i j\right>} s_i s_j - h \sum_i s_i
    $$
    $$
    \begin{align*}
        Z
        &= 
        \mathrm{Tr}
        \left(
        e^{-\beta \mathcal{H}}
        \right)
        \\&= 
        \sum_{\{s_i\}} 
        e^{-\beta \mathcal{H}\left(\{s_i\}\right)}
        \\&= 
        \sum_n^{2^N} 
        \left\langle n \right\vert
        e^{-\beta \mathcal{H}}
        \left\vert n \right\rangle
        \\&= 
        \sum_{n}^{2^N} 
        e^{-\beta E_n}
    \end{align*}
    $$
    Here $n$ is index of state, $\left\vert n \right\rangle$ is 
    ket state $n$, $N$ is total number of spins and $\{s_i\}$ means
    all possible configuration of Ising model.

    We cannot caliculate partition function directly except for 
    1D-lattice case and 2D-lattice case.
    However, by approximating Hamiltonian with mean field, we can 
    analytically obtain partition function.
    Let's approxiomate Hamiltonian.
    ### Ignoring high-order fluctuation
    First, let's replace $s_i$ with mean $\left< s_i \right>$ and 
    fluctuation from mean $\delta s_i = s_i - \left< s_i \right>$.
    $$
    \begin{align*}
        s_i 
        &= 
        \left< s_i \right> + \delta s_i
        \\&= 
        \left< s_i \right> 
        + \left( s_i - \left< s_i \right> \right)
    \end{align*}
    $$
    Here $\left< s_i \right>$ means
    $$
        \left< s_i \right>
        =
        \frac{1}{Z} \sum_{n=1}^{2^N} s_i \exp \left( -\beta E_n \right).
    $$
    $n$ is index of state 
    (total number of all state is $2^N$, $N$ is number of spin). 
    Inside of sum of first term of Hamiltonian becomes
    $$
    \begin{align*}
        s_i s_j
        &=
        \left( 
            \left< s_i \right> + \delta s_i
        \right) 
        \left( 
            \left< s_j \right> + \delta s_j
        \right) 
        \\&=
        \left< s_i \right> \left< s_j \right>
        + 
        \left< s_i \right> \delta s_j
        + 
        \delta s_i \left< s_j \right> 
        + 
        \delta s_i \delta s_j
        \\& \approx
        \left< s_i \right> \left< s_j \right>
        + 
        \left< s_i \right> \delta s_j
        + 
        \delta s_i \left< s_j \right> 
        \\&=
        \left< s_i \right> \left< s_j \right>
        + 
        \left< s_i \right> 
        \left( 
            s_j - \left< s_j \right>
        \right) 
        + 
        \left( 
            s_i - \left< s_i \right>
        \right) 
        \left< s_i \right> 
        \\&=
        \left< s_i \right> \left< s_j \right>
        + 
        \left< s_i \right> 
        \left( 
            s_j - \left< s_j \right>
        \right) 
        + 
        \left( 
            s_i - \left< s_i \right>
        \right) 
        \left< s_i \right> 
        \\&=
        -\left< s_i \right>^2
        + 
        \left< s_i \right> 
        \left( 
            s_i + s_j 
        \right) 
    \end{align*}
    $$
    We ignore the fluctuation with second order.
    We also used 
    $$
    \left< s_1 \right> 
    = \left< s_2 \right> 
    = \cdots
    = \left< s_i \right> 
    = \cdots
    = \left< s_N \right>
    $$
    because all spins are equivalent.

    What we need to notice is that magnetization in equilibrium state
    is equivalent to mean of spin $\left< s_i \right>$.
    $$
    \begin{align*}
        m 
        &= 
        \frac{1}{N} \sum_{i=1}^N \left< s_i \right>
        \\&= 
        \frac{1}{N} \left< s_i \right> \sum_{i=1}^N
        \\&= 
        \frac{1}{N} \left< s_i \right> N
        \\&= 
        \left< s_i \right>
    \end{align*}
    $$
    Thus we can replace $\left< s_i \right>$ with $m$.
    $$
        s_i s_j
        \approx 
        - m^2 + m(s_i + s_j)
    $$
    ### Mean-field Hamiltonian
    Then, mean-field Hamiltonian $\mathcal{H}_\mathrm{MF}$ beocomes
    $$
    \begin{align*}
        \mathcal{H}_\mathrm{MF}
        &= 
        -J \sum_{\left<i j\right>} 
        \left(- m^2 + m(s_i + s_j) \right) 
        - h \sum_i s_i
        \\&= 
        J m^2 \sum_{\left<i j\right>}  
        - J \sum_{\left<i j\right>} m(s_i + s_j)
        - h \sum_i s_i
    \end{align*}
    $$
    Let's think about first term.
    $$
    \begin{align*}
        J m^2 \sum_{\left<i j\right>}
        &=
        J m^2 \frac{1}{2} 
        \sum_{i} \sum_{\left<j\right>}
        \\&=
        J m^2 \frac{1}{2} 
        \sum_{i=1}^N z
        \\&=
        \frac{J N z}{2} m^2 
    \end{align*}
    $$
    Here $z$ is number of nearest-neighbor spins and division by 2 is
    for avoiding overlap.

    Move on to second term.
    $$
    \begin{align*}
        - J \sum_{\left<i j\right>} m(s_i + s_j)
        &=
        - J m 
        \left( 
            \sum_{\left<i j\right>} s_i + \sum_{\left<i j\right>} s_j
        \right)
        \\&=
        - 2 J m \sum_{\left<i j\right>} s_i
        \\&=
        - 2 J m \frac{1}{2} \sum_{i} s_i \sum_{\left<j\right>} 
        \\&=
        - J z m  \sum_{i} s_i
    \end{align*}
    $$
    Finally, mean-field Hamiltonian becomes
    $$
    \begin{align*}
        \mathcal{H}_\mathrm{MF}
        &= 
        \frac{J N z}{2} m^2 
        - J z m  \sum_{i} s_i
        - h \sum_i s_i
        \\&= 
        \frac{J N z}{2} m^2 
        - \left( J z m + h \right) \sum_i s_i
    \end{align*}
    $$
    ### Mean-field partition function
    We shut up and caliculate mean-field partition function.
    $$
        \begin{align*}
        Z_{\mathrm{MF}}
        &= 
        \sum_{\{s_i\}} 
        e^{-\beta \mathcal{H}\left(\{s_i\}\right)}
        \\&= 
        \sum_{s_1 = \pm 1} \sum_{s_2 = \pm 1} 
        \cdots \sum_{s_N = \pm 1}
        \exp 
        \left[
        -\beta 
        \mathcal{H} 
        \left( \{s_i\} \right)
        \right]
        \\&= 
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \exp 
        \left[
            - \beta\frac{J N z}{2} m^2 
            + \beta \left( J z m + h \right) \sum_i s_i
        \right]
        \\&= 
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \prod_{i=1}^N
        \exp 
        \left[
            \beta \left( J z m + h \right) s_i
        \right]
        \\&= 
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        \sum_{s_1 = \pm 1} 
        \exp 
        \left[
            \beta \left( J z m + h \right) s_1 
        \right]
        \right]
        \cdots 
        \left[
        \sum_{s_N = \pm 1}
        \exp 
        \left[
            \beta \left( J z m + h \right) s_N
        \right]
        \right]
        \\&= 
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        \sum_{s = \pm 1}
        \exp 
        \left[
            \beta \left( J z m + h \right) s
        \right]
        \right]^N
        \\&= 
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        2\cosh
        \left(
            \beta J z m + \beta h
        \right)
        \right]^N
        \end{align*}
    $$
    ### Self-consistent equation of magnetization
    We can also caliculate statistical mechanically averaged magnetization
    $m=\left<s_i\right>$.
    $$
    \begin{align*}
        m 
        &= \left<s_i\right>
        \\&=
        \frac{1}{Z_{\mathrm{MF}}}
        \sum_{s_1 = \pm 1} 
        \cdots 
        \sum_{s_N = \pm 1}
        s_i
        \exp 
        \left[
        -\beta 
        \mathcal{H} 
        \left( \{s_i\} \right)
        \right]
        \\&=
        \frac{1}{Z_{\mathrm{MF}}}
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        \sum_{s_1 = \pm 1} 
        \exp 
        \left[
            \beta \left(J z m + h \right) s_1 
        \right]
        \right]
        \cdots 
        \left[
        \sum_{s_i = \pm 1} 
        s_i
        \exp 
        \left[
            \beta \left(J z m + h \right) s_i
        \right]
        \right]
        \cdots 
        \left[
        \sum_{s_N = \pm 1} 
        \exp 
        \left[
            \beta \left(J z m + h \right) s_N
        \right]
        \right]
        \\&=
        \frac{1}{Z_{\mathrm{MF}}}
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        2\cosh 
        \left(
            \beta J z m + \beta h
        \right)
        \right]
        \cdots 
        \left[
        2\sinh
        \left(
            \beta J z m + \beta h
        \right)
        \right]
        \cdots 
        \left[
        2\cosh
        \left(
            \beta J z m + \beta h
        \right)
        \right]
        \\&=
        \frac
        {
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        2\cosh
        \left(
            \beta J z m + \beta h
        \right)
        \right]^{N-1}
        2\sinh
        \left(
            \beta J z m + \beta h
        \right)
        }
        {
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        2\cosh
        \left(
            \beta J z m + \beta h
        \right)
        \right]^N
        }
        \\&=
        \tanh
        \left(
            \beta J z m + \beta h 
        \right)
    \end{align*}
    $$
    When there is no external magnetic field $h=0$, this would be
    $$
        m = \tanh \left( \beta J z m \right)
    $$
    We got a self-consistent equation of magnetization.
    This form is analytically unsolvable but we can obtain $m$ from 
    graphical method. 
    By independetly plot the functions $y = m$ and 
    $y=\tanh \left( \beta J z m \right)$, we can see $m$ which satisfies 
    the self-consistent equation (consider $J$ and $z$ as constant).
    - make graph
    We clearly see that as the temperature increases, in a particular 
    temperature, number of solution become one from three.
    This qualitative change is phase transition.
    ### Critical temperature of mean-field approximation
    At which temperature, does this transition happen? 
    When the slope of $y=\tanh \left( \beta J z m \right)$ is same as 
    $y=m$ at $m=0$, these two equation has single intersection.
    Taylor series of $\tanh(x)$ at $x=0$ is $\tanh(x) \approx x$.
    Near $m=0$ we can say $\tanh(\beta Jzm) = \beta Jzm$.
    Then slope become
    $$
    \begin{align*}
        \frac{\mathrm{d}}{\mathrm{d}m} \tanh(\beta Jzm)
        &\approx
        \frac{\mathrm{d}}{\mathrm{d}m} \beta Jzm
        \\&=
        \beta Jz
    \end{align*}
    $$
    This is equal to the slope of $y=m$ at $m=0$ which is $1$.
    Then critical temperature $T_\mathrm{c}$ sarisfies
    $\frac{Jz}{k_\mathrm{B} T_\mathrm{c}} = 1$.
    Critical temperature $T_\mathrm{c}$ is 
    $$
        T_\mathrm{c}
        =
        \frac{Jz}{k_\mathrm{B}}.
    $$
    Critical temperature increases with $J$ and $z$ but it does not 
    depend on dimension of lattice.
    As we know, there is no phase transition in 1D-lattice Ising model.
    This results is qualitatively wrong in that case but as the dimension 
    become infinity this result become correct.
    ### Free energy of mean-field approximation
    We already have partition function. 
    Why don't we get free energy?
    $$
    \begin{align*}
        F_\mathrm{MF} 
        &= 
        - \frac{1}{\beta} \ln Z_\mathrm{MF}
        \\&=
        - \frac{1}{\beta} \ln  
        \left[
        \exp 
        \left(
            - \beta\frac{J N z}{2} m^2 
        \right)
        \left[
        2\cosh
        \left(
            \beta J z m + \beta h
        \right)
        \right]^N
        \right]
        \\&=
        \frac{JNz}{2}m^2
        - \frac{N}{\beta} \ln
        \left[
            2 \cosh \left(\beta Jzm + \beta h \right)
        \right]
    \end{align*}
    $$
    Notice that by differentiating free energy with magnetic field, we
    can obtain magnetization.
    $$
    \begin{align*}
        m &= \frac{1}{N}M
          \\&= 
          -
          \frac{1}{N} 
          \frac{\partial F_\mathrm{MF}}{\partial h}
          \\&= 
          \frac{1}{N} 
          \frac{N}{\beta}
          \frac
          {2 \sinh \left(\beta Jzm + \beta h \right)}
          {2 \cosh \left(\beta Jzm + \beta h \right)}
          \beta
          \\&=
          \tanh \left(\beta Jzm + \beta h \right)
    \end{align*}
    $$
    ### Critical exponent of mean-field approximation
    By introducint dimensionless temperature parameter $\theta$,
    $$
    \begin{align*}
     \theta 
     &= 
     \frac{T}{T_\mathrm{c}}
     \\&= 
     \frac{k_\mathrm{B}T}{Jz}
     \\&= 
     \frac{1}{\beta Jz}
    \end{align*}
    $$
    we can rewrite free energy as dimensionless form.
    $$
    \begin{align*}
        f_\mathrm{MF}
        &=
        \frac{F_\mathrm{MF}}{JzN}
        \\&= 
        \frac{m^2}{2}
        - \frac{1}{\beta Jz} 
        \ln
        \left[
            2 \cosh \left(\beta Jzm + \beta Jz h \frac{1}{Jz} \right)
        \right]
        \\&= 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta 
        \ln \cosh 
        \left( 
            \frac{1}{\theta} m 
             + \frac{1}{\theta} \frac{h}{Jz} 
        \right)
        \\&= 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta \ln \cosh 
        \left( 
            \frac{m + h'}{\theta} 
        \right)
    \end{align*}
    $$
    Here $h':=\frac{h}{Jz}$.
    To get intuitive idea of this free energy, let's use the Mclaurin 
    series expansion 
    $\cosh(x) = 1 + \frac{x^2}{2} + \frac{x^4}{24} + \mathcal{O}(x^6)$.
    In the case of $h'=0$ and $\frac{m}{\theta} \ll 1$, 
    free energy can be expanded as
    $$
        f_\mathrm{MF}
        = 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta \ln 
        \left[ 
            1
            +\frac{1}{2}\frac{m^2}{\theta^2}
            +\frac{1}{24}\frac{m^4}{\theta^4}
            + \mathcal{O}(\frac{m^6}{\theta^6})
        \right].
    $$
    Then we use the Mclaurin series expansion 
    $\ln(1+x) \approx x - \frac{x^2}{2}$
    $$
    \begin{align*}
        f_\mathrm{MF}
        &= 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta 
        \left[ 
            (\frac{1}{2}\frac{m^2}{\theta^2} 
            + \frac{1}{24}\frac{m^4}{\theta^4})
            - \frac{1}{2}
            (\frac{1}{2}\frac{m^2}{\theta^2} 
            + \frac{1}{24}\frac{m^4}{\theta^4})^2
            + \mathcal{O}\left(\frac{m^6}{\theta^6}\right)
        \right]
        \\&= 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta 
        \left[ 
            \frac{1}{2}\frac{m^2}{\theta^2} 
            + \frac{1}{24}\frac{m^4}{\theta^4}
            - \frac{1}{2}\frac{1}{4}\frac{m^4}{\theta^4} 
            + \mathcal{O}\left(\frac{m^6}{\theta^6}\right)
        \right]
        \\&= 
        \frac{m^2}{2}
        - \theta \ln 2
        - \theta 
        \left[ 
            \frac{1}{2}\frac{m^2}{\theta^2} 
            - \frac{1}{12}\frac{m^4}{\theta^4}
            + \mathcal{O}\left(\frac{m^6}{\theta^6}\right)
        \right]
        \\&= 
        \frac{1}{12}\frac{m^4}{\theta^3}
        + \frac{1}{2}m^2 \left( 1-\frac{1}{\theta} \right)
        - \theta \ln 2
        + \mathcal{O}\left(\frac{m^6}{\theta^6}\right)
    \end{align*}
    $$
    Let's check the extrema of this free energy.
    $$
    \begin{align*}
        0
        &=
        \frac{\partial f_\mathrm{MF}}{\partial m}
        \\&=
        \frac{1}{3}\frac{m^3}{\theta^3}
        + m \left( 1-\frac{1}{\theta} \right)
        \\&=
        m 
        \left(
        \frac{1}{3}\frac{m^2}{\theta^3}
        + \left( 1-\frac{1}{\theta} \right)
        \right)
    \end{align*}
    $$
    Except for $m=0$ this has a solution
    $$
    \begin{align*}
        m^2
        &=
        \left( \frac{1}{\theta} - 1 \right) \cdot 3\theta^3
        \\&=
        3\left( 1-\theta \right) \theta^2
        \\&=
        -3 t \theta^2.
    \end{align*}
    $$
    Here we introduced reduced temperature
    $$
    \begin{align*}
        t 
        &:= 
        \frac{T - T_\mathrm{C}}{T_\mathrm{C}}
        \\&= 
        \theta - 1
    \end{align*}
    $$
    Thus when $t<0$ i.e. $T<T_\mathrm{c}$, there are three local extrema 
    i.e. $m=0, \pm \sqrt{3} |t|^{1/2} \theta$.
    When $t>0$ i.e. $T>T_\mathrm{c}$, there is single local extrema $m=0$.
    From $m \sim (-t)^\beta \sim (-t)^{1/2}$, exponent of magnetization is
    $1/2$.
    
    To check these extrema are maxima or minima, we need to check second 
    derivative of free energy.
    $$
    \begin{align*}
        \frac{\partial^2 f_\mathrm{MF}}{\partial m^2}
        &=
        \frac{m^2}{\theta^3}
        + \left( 1-\frac{1}{\theta} \right)
        \\&=
        \frac{m^2}{(1+t)^3}
        + \left( \frac{1+t-1}{1+t} \right)
        \\&=
        \frac{m^2}{(1+t)^3}
        + \left( \frac{t}{1+t} \right)
        \\&=
        \frac{1}{1+t}
        \left[
        \frac{m^2}{(1+t)^2} + t
        \right]
    \end{align*}
    $$
    Thus when $t<0$ second derivative is positive for nonzero $m$?????
    ????????????????????

        """) 

    with st.sidebar:
        cols_sidebar =st.columns(2)
        size   = cols_sidebar[0].slider("size",0, 30, 10)
        J      = cols_sidebar[1].slider("J ",0.01, 2., 1.)
        nsteps = cols_sidebar[0].slider("Nsteps",0, 30, 10)
        beta   = cols_sidebar[1].slider("beta",0., 5., 1.) 
        cmap = st.select_slider('cmap', 
            ['inferno', 'gist_rainbow', 'RdBu', 'viridis',
            'inferno_r', 'magma'
            ])
        
    st.markdown(r"""
    ## 1D Ising model and transfer matrix method
    ### Hamiltonian 
    We can obtain partition function of 1D-lattice Ising model can 
    analytically.
    Hamiltonian of 1D-lattice Ising model is
    $$
        \mathcal{H}
        =
        -J\sum_{i=1}^{N} s_i s_{i+1} - h \sum_{i=1}^{N} s_i
    $$
    with periodic boundary condition $s_{N+1} = s_{1}$.
    ### Partition function
    Partition function becomes
    $$
    \begin{align*}
        Z 
        &=
        \sum_{\{s_i\}} 
        e^{-\beta \mathcal{H}\left(\{s_i\}\right)}
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \exp
        \left(
            \beta J\sum_{i=1}^{N} s_i s_{i+1} 
            + \beta h \sum_{i=1}^{N} s_i 
        \right)
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \exp
        \left[
            \beta J \left( s_1 s_2 + \cdots + s_N s_1 \right)
            + \beta h \left( s_1 + \cdots + s_N \right)
        \right]
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \exp
        \left(
            \beta J s_1 s_2 
            + \cdots
            + \beta J s_N s_1 
            + \beta h \frac{s_1+s_2}{2} 
            + \cdots
            + \beta h \frac{s_N+s_1}{2} 
        \right)
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \exp
        \left(
            \beta J s_1 s_2 
            + \beta h \frac{s_1+s_2}{2} 
        \right)
        \cdots
        \exp
        \left(
            \beta J s_N s_1 
            + \beta h \frac{s_N+s_1}{2} 
        \right)
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \prod_{i=1}^N
        \exp
        \left(
            \beta J s_i s_{i+1} 
            + \beta h \frac{s_i+s_{i+1}}{2} 
        \right)
    \end{align*}
    $$
    ### Transfer matrix
    $$
        T_{s_i, s_{i+1}} 
        =
        \exp
        \left(
            \beta J s_i s_{i+1} 
            + \beta h \frac{s_i+s_{i+1}}{2} 
        \right)
    $$
    is a element of transfer matrix $T$. Transfer matrix is 
    $$
    \begin{align*}
        T
        &=
        \begin{bmatrix}
        T_{+1, +1} & T_{+1, -1} \\
        T_{-1, +1} & T_{-1, -1}
        \end{bmatrix}
        \\&=
        \begin{bmatrix}
        \exp \left[ \beta(J+h) \right] & \exp \left[ -\beta J \right] \\
        \exp \left[ -\beta J \right] & \exp \left[ \beta(J-h) \right]
        \end{bmatrix}
    \end{align*}
    $$
    From the definition of matrix multiplication, we see
    $$
        \left(T^2\right)_{s_i, s_{i+2}} 
        =
        \sum_{s_{i+1} = \pm1}
        T_{s_i, s_{i+1}}
        T_{s_{i+1}, s_{i+2}}
    $$
    Using these let's caliculate partition function!
    $$
    \begin{align*}
        Z 
        &=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        \prod_{i=1}^N
        T_{s_i, s_{i+1}}
        \\&=
        \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
        T_{s_1, s_2}
        \cdots
        T_{s_N, s_1}
        \\&=
        \sum_{s_1 = \pm 1} 
        \sum_{s_3 = \pm 1} 
        \cdots
        \sum_{s_N = \pm 1} 
        \left(
        \sum_{s_2 = \pm 1} 
        T_{s_1, s_2}
        T_{s_2, s_3}
        \right)
        T_{s_3, s_4}
        \cdots
        T_{s_N, s_1}
        \\&=
        \sum_{s_1 = \pm 1} 
        \sum_{s_3 = \pm 1} 
        \cdots
        \sum_{s_N = \pm 1} 
        \left(T^2\right)_{s_1, s_3}
        T_{s_3, s_4}
        \cdots
        T_{s_N, s_1}
        \\&=
        \sum_{s_1 = \pm 1} 
        \cdots
        \sum_{s_N = \pm 1} 
        \left(
        \sum_{s_3 = \pm 1} 
        \left(T^2\right)_{s_1, s_3}
        T_{s_3, s_4}
        \right)
        T_{s_4, s_5}
        \cdots
        T_{s_N, s_1}
        \\&= 
        \cdots
        \\&= 
        \sum_{s_1 = \pm 1} 
        \left(
        \sum_{s_N = \pm 1} 
        \left(T^{N-1}\right)_{s_1, s_N}
        T_{s_N, s_1}
        \right)
        \\&= 
        \sum_{s_1 = \pm 1} 
        \left(T^{N}\right)_{s_1, s_1}
        \\&= 
        \mathrm{Tr}
        \left(
        T^{N}
        \right)
    \end{align*}
    $$
    Here $T$ is a real symmetric matrix which is diagonalizable.
    $$
        T = PDP^{-1}
    $$
    where $D$ is a diagonal matrix with eigenvalues $\lambda_1$ and 
    $\lambda_2$
    $$
        D 
        = 
        \begin{bmatrix}
        \lambda_1 & 0 \\
        0 & \lambda_2
        \end{bmatrix}
    $$
    and $P$ is a matrix containing eigenvectors.
    Thanks to this diagonalization $T^{N}$ becomes simpler.
    $$
    \begin{align*}
        T^{N} 
        &=
        PDP^{-1}PDP^{-1} \cdots PDP^{-1}
        \\&=
        PD^{N}P^{-1}
    \end{align*}
    $$
    Using the property of trace, 
    $$
    \begin{aligned}
        \operatorname{Tr}(A B) &=\sum_i(A B)_{i i} \\
        &=\sum_i \sum_j A_{i j} B_{j i} \\
        &=\sum_j \sum_i B_{j i} A_{i j} \\
        &=\sum_j(B A)_{j j} \\
        &=\operatorname{Tr}(B A)
    \end{aligned}
    $$
    we obtain
    $$
    \begin{aligned}
        \mathrm{Tr}
        \left(
            PD^{N}P^{-1}
        \right)
        &=
        \mathrm{Tr}
        \left(
            D^{N}P^{-1}P
        \right)
        \\&=
        \mathrm{Tr}
        \left(
            D^{N}
        \right)
    \end{aligned}
    $$
    Returning to partition function
    $$
    \begin{aligned}
        Z
        &=
        \mathrm{Tr}
        \left(
            T^{N}
        \right)
        \\&=
        \mathrm{Tr}
        \left(
            D^{N}
        \right)
        \\&=
        \lambda_1^N + \lambda_2^N
    \end{aligned}
    $$
    Cool. We need eigenvalue of transfer matrix.
    $$
    \begin{aligned}
        0
        &=
        \operatorname{det}(T-\lambda)
        \\&=
        \left|\begin{array}{cc}
        \exp\left({\beta J+ \beta h}\right)-\lambda & \exp\left({-\beta J}\right) \\
        \exp\left(-\beta J\right) & \exp\left(\beta J - \beta h\right)-\lambda
        \end{array}\right|
        \\&=
        \left(\exp\left(\beta J+ \beta h\right)-\lambda\right)
        \left(\exp\left(\beta J- \beta h\right)-\lambda\right)
        -\exp\left(-2 \beta J\right)
        \\&=
        \exp\left(2\beta J\right)
        -\lambda
        \left(
            \exp\left(\beta J + \beta h\right)
            +
            \exp\left(\beta J - \beta h\right)
        \right)
        +\lambda^2
        -\exp\left(-2 \beta J\right)
    \end{aligned}
    $$
    We obtaiend quadratic equation of $\lambda$.
    $$
    \begin{aligned}
        0
        &=
        \lambda^2
        - 
        \lambda
        \left[
             \exp\left(\beta J+ \beta h\right)
            +\exp\left(\beta J- \beta h\right)
        \right]
        + 
        \left[
        \exp\left(2\beta J\right)
        - \exp\left(-2 \beta J\right)
        \right]
        \\&=
        \lambda^2
        - 
        2 \lambda \exp \left(\beta J\right) \cosh \left(\beta h\right)
        + 
        2 \sinh \left(2 \beta J\right)
    \end{aligned}
    $$
    This equation has two solutions.
    $$
    \begin{aligned}
        \lambda 
        &=
        \exp \left(\beta J\right) \cosh \left(\beta h\right)
        \pm
        \sqrt{
            \exp \left(2 \beta J\right) \cosh^2 \left(\beta h\right)
            -
            2 \sinh \left(2 \beta J\right)
        }
        \\&=
        \exp \left(\beta J\right) \cosh \left(\beta h\right)
        \pm
        \sqrt{
            \exp \left(2 \beta J\right) 
            +
            \exp \left(2 \beta J\right) 
            \sinh^2 \left(\beta h\right)
            -
            \left(
                \exp\left(2 \beta J\right)
                +
                \exp\left(-2 \beta J\right)
            \right)
        }
        \\&=
        \exp \left(\beta J\right) \cosh \left(\beta h\right)
        \pm
        \sqrt{
            \exp \left(2 \beta J\right) 
            \sinh^2 \left(\beta h\right)
            -
            \exp\left(-2 \beta J\right)
        }
    \end{aligned}
    $$






    """)   
    
    
    chain = np.zeros(size) ; chain[chain<.5] = -1; chain[chain>=.5] = 1
    CHAINS = []
    for _ in range(nsteps):
        # pick random site
        i = np.random.randint(0,size-1)
        dE = (sum(chain[i-1:i+2])-chain[i])*chain[i]
        if np.random.rand()<np.exp(-beta*dE):
            chain[i] *= -1
        CHAINS.append(chain.copy())

    
    CHAINS = np.array(CHAINS)
    fig, ax = plt.subplots()
    ax.imshow(CHAINS, cmap=cmap, aspect = size/nsteps/3)
    ax.set_ylabel('Timestep', color='white')
    ax.set_xlabel('Site index', color='white')
    st.pyplot(fig)


    st.markdown(r"""
    ## Transfer Matrix Method
    ...
    """)

def run_percolation_and_fractals():
    # Side bar
    with st.sidebar:
        st.markdown('## Paramteres') 
        with st.expander('square grid percolation'):

            cols_sidebar = st.columns(2)
            size = cols_sidebar[0].slider('size', 10  , 100, 50)
            p = cols_sidebar[1].slider('p',       0.01, 1. , .1)
            marker_dict = {
                'point': '.',
                'square': 's',
                'pixel': ',',
                'circle': 'o',
            }
            marker_key = st.select_slider('marker', marker_dict.keys())
            marker = marker_dict[marker_key]
            seed = st.slider('seed',10,100)

        with st.expander('Bethe lattice'):

            cols_sidebar = st.columns(2)
            levels = cols_sidebar[0].slider('levels', 0  , 3, 2)
            p = cols_sidebar[1].slider('p_',       0.01, 1. , .1)

            

        with st.expander('Mandelbrot'):
            cols_sidebar = st.columns(2)
            logsize = cols_sidebar[0].slider(r'Resolution (log)',1.5,4., 3.)
            size_fractal = int(10**logsize)
            cols_sidebar[1].latex(r'10^{}\approx {}'.format("{"+str(logsize)+"}", size))
            cols_sidebar = st.columns(2)
            n = cols_sidebar[0].slider('n',1,50,27)
            a = cols_sidebar[1].slider('a',0.01,13.,2.3)

    # Functions
    def makeGrid(size, seed=42): 
        np.random.seed(seed)
        grid = np.random.uniform(0,1,(size,size), )
        grid_with_border = np.ones((size+2,size+2))
        grid_with_border[1:-1, 1:-1] = grid
        return grid_with_border

    def checkNeighbours(pos, grid, domain, visited):
        (i,j) = pos
        neighbours = [(i-1,j), (i+1,j), (i,j-1), (i, j+1)]
        for n in neighbours:
            if (n[0]>=0) and (n[1]>=0) and (n[0]<len(grid)) and (n[1]<len(grid)):
                if grid[n] and (n not in visited):
                    domain.add(n)
                    visited.add(n)
                    domain_, visited_ = checkNeighbours(n, grid, domain, visited)
                    domain = domain.union(domain_)
                    visited = visited.union(visited_)
                else: visited.add(n)
        return domain, visited

    def getDomains(grid, p=.5):
        open_arr = grid < p
        domains = {} ; index = 0; visited = set()
        for i, _ in enumerate(open_arr):
            for j, val in enumerate(open_arr[i]):
                if val:
                    if (i,j) in visited:
                        domain, visited_ = checkNeighbours((i,j), open_arr, domain=set(), visited=visited)
                    else:
                        visited.add((i,j))
                        domain, visited_ = checkNeighbours((i,j), open_arr, domain=set([(i,j)]), visited=visited)
                    domains[index] = domain
                    visited = visited.union(visited_)
                    index+=1
                else:
                    visited.add((i,j))
        
        new_domains = {}
        index = 0
        for d in domains:
            if len(domains[d]) !=0:
                new_domains[index] = domains[d]
                index += 1
                
        return new_domains

    def percolation():
        grid = makeGrid(size,seed)
        domains = getDomains(grid, p)

        x = np.arange(size+2)
        X,Y = np.meshgrid(x,x)
        
        fig, ax = plt.subplots()
        # background
        ax.scatter(X,Y, c='black')

        # colors
        colors = sns.color_palette("hls", len(domains))
        np.random.shuffle(colors)
        colors = np.concatenate([[colors[i]]*len(domains[i]) for i in domains])

        # plot
        xx = np.concatenate([list(domains[i]) for i in domains])
        ax.scatter(xx[:,0], xx[:,1], c=colors, marker=marker)
        ax.set(xticks = [], yticks = [], facecolor='black')
        return fig

    def betheLattice():
        # Create a graphlib graph object
        graph = graphviz.Digraph()

        root = str(0)
        nodes = []
        for other in '0 1 2'.split():
            graph.edge(root, root+other)
            nodes.append(root+other)

        new_nodes = []
        for i in nodes:
            for j in range(2):
                graph.edge(str(i), str(i)+str(j))

                new_nodes.append(str(i)+str(j))

        nodes = new_nodes
        new_nodes = []
        for i in nodes:
            for j in range(2):
                graph.edge(str(i), str(i)+str(j))
                new_nodes.append(str(i)+str(j))
        return graph


    def run_fractals():
        def stable(z):
            try:
                return False if abs(z) > 2 else True
            except OverflowError:
                return False
        stable = np.vectorize(stable)


        def mandelbrot(c, a, n=50):
            z = 0
            for i in range(n):
                z = z**a + c
            return z

        def makeGrid(resolution, lims=[-1.85, 1.25, -1.25, 1.45]):
            re = np.linspace(lims[0], lims[1], resolution)[::-1]
            im = np.linspace(lims[2], lims[3], resolution)
            re, im = np.meshgrid(re,im)
            return re+im*1j

        def plot_(res):
            fig = plt.figure(figsize=(12,6))
            plt.imshow(res.T, cmap='magma')
            plt.xticks([]); plt.yticks([])
            plt.xlabel('Im',rotation=0, loc='right', color='blue')
            plt.ylabel('Re',rotation=0, loc='top', color='blue')
            return fig

        res = stable(mandelbrot(makeGrid(size_fractal,  lims=[-1.85, 1.25, -1.25, 1.45]), a=a, n=n))
        return plot_(res)

    # Render
    st.markdown(r"""# Percolation and Fractals""")

    st.markdown(r"""## Percolation""")
    st.pyplot(percolation())
    

    st.markdown(r"""
    A matrix containing values between zero and one, with
    the value determining openness as a function of $p$.

    After generating a grid and a value for p, we look for 
    connected domains. 
    """)

    st.markdown(r"""
    ## Bethe Lattice
    Bethe lattice (also called a regular tree)  is an infinite connected 
    cycle-free graph where all vertices have the same number of neighbors.  
    """)
    
    st.graphviz_chart(betheLattice())

    st.markdown(r"## Percolation on this lattice")


    st.pyplot(run_fractals())


    st.markdown(r"""
    The Mandelbrot set contains complex numbers remaining stable through
    
    $$z_{i+1} = z^a + c$$
    
    after successive iterations. We let $z_0$ be 0.
    """)
    st.code(r"""
    def stable(z):
        try:
            return False if abs(z) > 2 else True
        except OverflowError:
            return False
    stable = np.vectorize(stable)


    def mandelbrot(c, a, n=50):
        z = 0
        for i in range(n):
            z = z**a + c
        return z

    def makeGrid(resolution, lims=[-1.85, 1.25, -1.25, 1.45]):
    re = np.linspace(lims[0], lims[1], resolution)[::-1]
    im = np.linspace(lims[2], lims[3], resolution)
    re, im = np.meshgrid(re,im)
    return re+im*1j    """)

def run_random_walk():
    # Sidebar
    with st.sidebar:
        cols_sidebar = st.columns(2)
        nsteps = cols_sidebar[0].slider('nsteps',  4,   100, 14)
        seed   = cols_sidebar[1].slider('Seed',    0,   69 , 42)
        sigma2 = cols_sidebar[0].slider('Variance',0.2, 1. ,0.32)
        step_size = cols_sidebar[0].slider('Stepsize = random^x, x=', 0.,3.,0.)
        axisscale = cols_sidebar[1].radio('axis-scales', ['linear', 'symlog'])
        #yscale = cols_sidebar[1].radio('yscale', ['linear', 'symlog'])
    
    # Functions
    def accumulate(x):
        X=np.zeros(len(x)) ; X[0] = x[0]
        for i, _ in enumerate(x): X[i] = X[i-1]+x[i]
        return X

    def randomWalk(nsteps, sigma2=1, seed=42, axisscale='linear', step_size=0):
        (dx_f, dy_f) = (lambda theta, r=1: r*trig(theta) for trig in (np.cos, np.sin)) 
        dx_f = lambda theta, r = 1: r*np.cos(theta)
        dy_f = lambda theta, r = 1: r*np.sin(theta)

        np.random.seed(seed)
        thetas_uniform = np.random.uniform(0,2*np.pi,nsteps)
        thetas_randn = np.random.randn(nsteps)*sigma2
        thetas_bimodal = np.concatenate([ np.random.randn(nsteps//2) * sigma2-1, 
                                          np.random.randn(nsteps//2) * sigma2+1 ])

        thetas_uniform[0], thetas_randn[0], thetas_bimodal[0] = 0, 0, 0 

        rands = [thetas_uniform, thetas_randn, thetas_bimodal]
        rands_names = 'uniform, normal, bimodal'.split(', ')
        stepLengths = np.random.rand(nsteps).copy()**step_size

        def plot2():
            colors = 'r g y'.split()
            fig = plt.figure(figsize=(12,6))
            
            gs = GridSpec(3, 3, figure=fig)
            ax1 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
            ax2 = fig.add_subplot(gs[:, 1:])    
            
            lims = {'xl':0, 'xh':0, 'yl':0, 'yh':0}
            for i, (r, n, stepLength) in enumerate(zip(rands, rands_names, stepLengths)):
                dx, dy = dx_f(r, stepLength), dy_f(r, stepLength)
                dx = np.vstack([np.zeros(len(dx)), dx]).T.flatten()
                dy = np.vstack([np.zeros(len(dy)), dy]).T.flatten()
                
                ax1[i].plot(dx,dy, lw=1, c=colors[i])
                ax1[i].set(ylim=(-1,1), xlim=(-1,1), 
                            xticks=[], yticks=[],facecolor = "black",)
                
                x = accumulate(dx)
                y = accumulate(dy)
                ax2.plot(x,y, lw=2, label=n,c=colors[i])

            ax2.set(facecolor = "black",# xticks=[], yticks=[],
                    xticklabels=[],
                    xscale=axisscale,
                    yscale=axisscale)
            ax2.legend(fontsize=20)
                
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax1[0].set_title('Individual steps', fontsize=24)
            ax2.set_title('Cumulative path', fontsize=24)
            plt.tight_layout()
            fig.patch.set_facecolor('darkgrey')  # do we want black or darkgrey??
            return fig
        return plot2()



    st.markdown(r"""# RandomWalk""")
    st.pyplot(randomWalk(nsteps,sigma2, seed, axisscale, step_size))

    cols = st.columns(2)
    cols[0].markdown(r"""
    Continous steps in a random direction illustrates the
    differences between diff. distributions.

    Red steps are generated by sampling theta from a uniform distribution.
    This tends to keep us close to the origin.

    Normal and bi-modal distributions are different in that the
    similarity of step direction causes great displacement.
    """)

    cols[1].code(r"""
def randomWalk(nsteps):
    for i in range(nsteps):
        theta = random()
        dx = np.cos(theta) ; x += dx
        dy = np.sin(theta) ; y += dy 
    """)

    st.markdown(r"""
    ## First return
    *Explore the time of first return in 1d, 2d and 3d*
        """)

    # 1d
    def run_firstReturn1D():
        lengths = []
        lines = {}
        c=st.empty()
        for idx in range(100):
            

            x = [0] 
            for i in range(100):
                change = -1 if np.random.rand()< 0.5 else 1
                x.append(x[i]+change)
                if x[i+1] == 0: break
            lines[idx] = x

            fig, ax = plt.subplots(1,2)
            for idx in lines.keys():
                x = lines[idx]
            
                ax[0].plot(x, range(len(x)))#, c='orange')
            ax[0].set_xlabel('x position', color='white')
            ax[0].set_ylabel('time', color='white')
            ax[0].set(xticks=[0], yticks=[])
            ax[0].grid()

            lengths.append(len(x))

            ax[1].hist(lengths)
            ax[1].set_xlabel('First return time', color='white')
            ax[1].set_ylabel('occurance frequency', color='white')
            #ax[1].set(xticks=[0], yticks=[])
            ax[1].grid()
            c.pyplot(fig)

    a = st.button('run_firstReturn1D')
    if a: run_firstReturn1D()




def newNetwork():
    def makeBetheLattice(n_nodes = 10):
        M = np.zeros((n_nodes,n_nodes))

        idx = 1
        for i, _ in enumerate(M):
            if i ==0: n =3
            else: n =2
            M[i, idx:idx+n] = 1
            idx+=n

        return M+M.T

    import seaborn as sns

    def checkOpenNeighbours(open_neighbours,visited, domain, open_arr):
        
        for j in open_neighbours:
            if j not in visited:
                domain.append(j)
                visited.add(j)
                open_neighbours = np.argwhere(M[j] * open_arr).flatten()
                open_neighbours = open_neighbours[open_neighbours!=j]
                visited_, domain_ = checkOpenNeighbours(open_neighbours,visited, domain,open_arr)
                visited = visited.union(visited_)
                domain += domain
        return visited, domain
        
    def getDomains(M, p=0.3):
        
        open_arr = p>np.random.rand(len(M))
        visited = set()
        domains = []
        for i in range(len(M)):

            if i not in visited:
                if open_arr[i]:
                    domain = []
                    visited.add(i)
                    domain.append(i)

                    open_neighbours = np.argwhere(M[i] * open_arr).flatten()
                    open_neighbours = open_neighbours[open_neighbours!=i]
                    if len(open_neighbours)>0:
                        visited_, domain_ = checkOpenNeighbours(open_neighbours,visited, domain,open_arr)
                        visited = visited.union(visited_)
                        domain += domain
                    domains.append(set(domain))

        return domains
                    
                


    def draw_from_matrix(M, domains) :
        
        inDomain = {}
        for idx, d in enumerate(domains):
            for i in d:
                inDomain[i] = idx
        inDomain   
        
        
        G = nx.Graph()
        for i, line in enumerate(M):
            G.add_node(i)

        for i, line in enumerate(M):
            for j, val in enumerate(line):
                if (i != j) and (val==1): 
                    G.add_edge(i, j)
        palette = sns.color_palette('hls', len(domains))
        color_map = ['darkgrey' if i not in inDomain.keys() else palette[inDomain[i]] for i in range(len(M))]

        nx.draw_networkx(G, node_color=color_map, pos=nx.kamada_kawai_layout(G))
        
        
    import networkx as nx
    import numpy as np
    M = makeBetheLattice(34)
    domains = getDomains(M,0.6)
    open_arr = draw_from_matrix(M,domains)


def bereaucrats():
    st.markdown(r"# Beraucrats")


    def makeGrid(size):
        arr = np.zeros((size,size))
        return arr, size**2
    
    def fill(arr, N):
        rand_index = np.random.randint(0,N)
        who = (rand_index//arr.shape[1], rand_index%arr.shape[1])
        arr[who] += 1
        return arr


    def run():
        arr, N = makeGrid(size)
        results = {'mean':[], 'arr':[]}
        
        for step in range(nsteps):
            arr = fill(arr, N)  # bring in 1 task

            overfull_args = np.argwhere(arr>=4)  # if someone has 4 tasks redistribute to neighbours
            for ov in overfull_args:
                (i,j) = ov
                for pos in [(i+1, j), (i-1, j), (i,j+1), (i,j-1)]:
                    try: arr[pos] +=1
                    except: pass
                arr[i,j] -= 4
            results['mean'].append(np.mean(arr)) 
            results['arr'].append(arr.copy()) 

        return results

    with st.sidebar:
        cols_sidebar = st.columns(2)
        size = cols_sidebar[0].slider(r'size',5,40, 10)
        nsteps = cols_sidebar[1].slider('nsteps',1,5000,1000)
    
    results = run()

    # plot 1
    c = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart = st.line_chart()
    fig, ax = plt.subplots(1,5, figsize=(12,2.5))
    a = [ax[idx].set(xticks=[], yticks=[], 
                facecolor='black') for idx in range(5)]
    
    steps = range(nsteps)[::nsteps//10]
    
    for val, (i, next) in enumerate(zip(steps, steps[1:])):
        
        progress_bar.progress(int((i + 1)/nsteps))  # Update progress bar.

        if val%2==0:  # plot imshow the grid
            idx = 0 if val==0 else idx+1 
            ax[idx].imshow(results['arr'][i], cmap="inferno")
            ax[idx].set(xticks=[], yticks=[])
            c.pyplot(fig)
            
        
        new_rows = results['mean'][i:next]

        # Append data to the chart.
        chart.add_rows(new_rows)

        # Pretend we're doing some computation that takes time.
        time.sleep(.1)

    status_text.text('Done!')

    st.markdown(r"""
    The problem with beraucrats, is that they dont finish tasks. When a task 
    lands on the desk of one, the set a side to start a pile. When that pile contains 
    4 tasks, its time to distribute them amongst the nieghbors. If a
    beraucrat neighbours an edge, they may dispose of the task headed in that direction. 
    """)


def bakSneppen():
    st.markdown(r"# Bak-Sneppen")
    def run(size, nsteps):
        chain = np.random.rand(size)

        X = np.empty((nsteps,size))
        L = np.zeros(nsteps)
        for i in range(nsteps):
            lowest = np.argmin(chain)  # determine lowest
            chain[(lowest-1+size)%size] = np.random.rand() # change left neighbour
            chain[lowest] = np.random.rand() # change ego
            chain[(lowest+1)%size] = np.random.rand() # change right neighbour
            X[i] = chain
            L[i] = np.mean(chain)

        fig, ax = plt.subplots()
        ax.imshow(X, aspect  = size/nsteps, vmin=0, vmax=1, cmap='gist_rainbow')
        st.pyplot(fig)
        return L

    with st.sidebar:
        nsteps = st.slider('nsteps',1,30000,5000)
        size = st.slider('size',10,1000,300)

    L = run(size, nsteps)
    cols = st.columns(2)

    cols[0].markdown(r"""
    The Bak-Sneppen method starts with a random vector. At each
    timestep the smallest element and its two neighbors, are each 
    replaced with new random numbers.

    The figure on the right depicts the mean magnitude of elements in
    the vector.

    To build further on this, we should identify power laws along each dimension.
    """)

    fig, ax = plt.subplots()
    ax.plot(range(len(L)), L, c='purple')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(facecolor=(.04,.065,.03))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    cols[1].pyplot(fig)


def network():

    def makeBetheLattice(n_nodes = 10):
        M = np.zeros((n_nodes,n_nodes))

        idx = 1
        for i, _ in enumerate(M):
            if i ==0: n =3
            else: n =2
            M[i, idx:idx+n] = 1
            idx+=n

        return M+M.T

    def make_network(n_persons = 5,alpha=.4):
        
        A = np.zeros((n_persons,n_persons))
        for i in range(n_persons):
            neighbours =  np.random.rand(n_persons)>alpha ; neighbours[i]=0
            if sum(neighbours) == 0: 
                a = np.random.randint(0,n_persons,4)
                a = a[a!=i][0]
                neighbours[a] =1
            A[i] += neighbours; A[:,i] = neighbours
        
        return A

    def draw_from_matrix(M, sick=[], pos=[]):
        sick = np.zeros(len(M)) if len(sick) == 0 else sick
        G = nx.Graph()
        for i, line in enumerate(M):
            G.add_node(i)

        for i, line in enumerate(M):
            for j, val in enumerate(line):
                if (i != j) and (val==1): 
                    G.add_edge(i, j)
        color_map = ['r' if s==1 else 'white' for s in sick]
        
        pos = nx.nx_agraph.graphviz_layout(G) if len(pos)==0 else pos
        
        nx.draw_networkx(G,pos, node_color=color_map, edge_color='white')
        return pos


    with st.sidebar:
        network_type = st.selectbox('networt_type',['bethe', 'random'])
        N = st.slider('N',1,42,22)
        if network_type == 'random':
            a = st.slider('alpha', 0.,1.,0.97)
        
    fig, ax = plt.subplots()
    net = make_network(N,a) if network_type == 'random' else makeBetheLattice(N)
    draw_from_matrix(net)
    st.pyplot(fig)


def run_betHedging():

    st.markdown('# Bet-Hedghing')
    with st.sidebar:
        cols_sidebar = st.columns(2)
        nsteps = cols_sidebar[0].slider('nsteps',1,3000,500)
        starting_capital = cols_sidebar[1].slider('starting capital',1,1000,10)
        prob_loss = cols_sidebar[0].slider('loss probability', 0.,1.,.5) 
        invest_per_round = cols_sidebar[1].slider('invest per round', 0.,1.,.5) 

    capital = [starting_capital]
    for i in range(nsteps):
        if np.random.uniform()>prob_loss:
            capital.append(capital[i]*(1+invest_per_round))
        else:
            capital.append(capital[i]*(1-invest_per_round))

    fig, ax = plt.subplots()
    plt.plot(capital, c='purple')
    plt.xlabel('timestep', color='white')
    fig.patch.set_facecolor((.04,.065,.03))
    ax.set(yscale='log')
    plt.ylabel('capital', color='white')
    st.pyplot(fig)


func_dict = {
    'Statistical Mechanics' : run_stat_mech,
    'Phase transitions & Critical phenomena' : run_phaseTransitions_CriticalPhenomena,
    'Percolation and Fractals'   : run_percolation_and_fractals,
	'RandomWalk'    : run_random_walk,
    'Bereaucrats'   : bereaucrats,
    'Bak-Sneppen'   : bakSneppen,
    #'new network'   : newNetwork,
    #'Networks'      : network,
    'Bet-Hedghing'  : run_betHedging,
    #'Bethe Lattice' : run_betheLattice
}

with st.sidebar:
	topic = st.selectbox("topic" , list(func_dict.keys()))

a = func_dict[topic] ; a()


#plt.style.available
