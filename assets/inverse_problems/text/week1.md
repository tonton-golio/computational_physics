
# Header 1

*Analysis of indirect physical measurements.* I.e., what generates the observed data.

$$
	\begin{align*}
		\text{data} &= \text{function}(\text{parameters})\\
		d &= g(m)
	\end{align*}
$$

we know $d$ and usually we know $g$. If the generating function, $g$, were invertible, we could simply take $m = g^{-1}(m)$, but this is rarely the case.


Problems arising from trying to solve an inverse problem:
* solution may not exist $\rightarrow$ *look for **almost** solution*.
* Solution may not be unique $\rightarrow$ *consider physical permissability*.
* Solution does not depend continously on $d$, i.e., instability $\Rightarrow$ think phase-transitions.


Inverse problems **typically** have at least one of these challenges.


# Examples
#### Lunar tomography
*What does the inside of the moon look like?*

Moon-quakes measured at different points gives us different arrival times. This tells us about the wave-speeds, and we can thus say something about the internal structure of the moon...


Don't know if this is really relevant...??

#### Seismic tomography
We can do the same thing on earth...


#### Acoustice Waveform inversion
$$
	\frac{1}{\kappa(x)}\frac{\partial^2p}{\partial t^2}(x,t) - \nabla\cdot\left(
	\frac{1}{\rho(x)}\nabla p(x,t)
	\right)
	=
	S(x,t)
$$

$$
	\begin{align*}
		\text{Source:}&  p(x_0,t)\\
		\text{Boundaray:}&  p(x,t)=0\\
		\text{Data:}&  p(x_n,t) \text{for} n=1, \ldots, N
	\end{align*}
$$

**Inverse problem**:
Find acceptable $\kappa(x)$ and $\rho(x)$ such the wave equation satisfies initial and boundary conditions and reproduces the data.


**examples**:

* How does sound in a concert hall bounce off the walls?
* Ultrasound scanning





#### Can one hear the shape of a drum? 🧐

answer: yes, but is it unique? --> can two differently shaped drums make the same sound?

(insert pic of non-trivial drum)

Membrane is fixed at the boundary, i.e,. $u_\text{boundary}=0$

data = eigen frequencies 
unknown = shape of drum, i.e. $\delta D$

OMG, the answer is no!!! This was proben by Gordon Webb and Wolpert in 1991. They showed that two different shapes have the same eigenfrequencies.


So What can we hear??
Area, circumference and number of holes...



#### Earth's Eigenfreqs.
Big earthquakes make the earth wobble in manner similar to that of a baseball
(insert baseball gif.)


#### Can one hear the length of a string?
Yes 😉


#### Hadamard Heat-flow
Hadamard worked on a heat-flow problem (2d). Use diffusion:

$$
	\begin{align*}
		T(x,0) &= 0\\
		\frac{\partial T}{\partial y} &= \frac{1}{n}\sin(nx)\\
		\nabla^2 &= 0
	\end{align*}
$$

Yields the solution:
$$
	T(x,y) = \frac{1}{n^2}\sin(nx)\sinh(nx)
$$

let's explore the limits:

for $y=0$,  $\frac{\partial T}{\partial y} \rightarrow 0$ for $n\rightarrow\infty$ so no problem

for $y>0$,  $\text{sup}|T(x,y)| \rightarrow \infty$ for $n\rightarrow\infty$ so big problem. 

When you go deep (with Kassem G) we have a divergence! OMG this is problem number 3 i.e., instability.


**Hadamard called this an ill-posed problem.**


# Header 2

Inserting the prior on a plot (d versus m) and then inserting the function $d=g(m)$, the insertion of the function and the prior yields the posterior.


the posterior probability density of $\sigma(m)$:
$$
	\sigma(m) = \rho_m(m) L(m)
$$
where, $L(m) = \rho_d(g(m))$


If we have a high dimenisonal probability manifold, we may sample this manifold using Monte-Carlo.




