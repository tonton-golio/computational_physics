
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
    &= 
    \sum_{\{s_i\}} 
    e^{-\beta \mathcal{H}\left(\{s_i\}\right)}
    &= 
    \sum_n^{2^N} 
    \left\langle n \right\vert
    e^{-\beta \mathcal{H}}
    \left\vert n \right\rangle
    &= 
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