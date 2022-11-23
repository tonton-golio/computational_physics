
# Header 1

*Analysis of indirect physical measurements.* I.e., what generates the observed data.

$$
	\begin{align*}
		\text{data} &= \text{function}(\text{parameters})\\
		d &= g(m)
	\end{align*}
$$

we know $d$ and usually we know $g$. If the generating function, $g$, were invertible, we could simply take $m = g^{-1}(m)$, but this is rarely the case.

# Header 2
Problems arise in trying to solve an inverse problem:
* solution may not exist $\rightarrow$ *look for **almost** solution*.
* Solution may not be unique $\rightarrow$ *consider physical permissability*.
* Solution does not depend continously on $d$, i.e., instability $\Rightarrow$ think phase-transitions.


# Examples
#### Examples
##### Lunar tomography
*What does the inside of the moon look like?*

Moon-quakes measured at different points gives us different arrival times. Density variations in the bulk of the moon, affect wave-speeds. Thus we may predict characteristics of the internal structure of the moon.

##### Acoustice Waveform inversion
$$
	\frac{1}{\kappa(x)}\frac{\partial^2p}{\partial t^2}(x,t) - \nabla\cdot\left(
	\frac{1}{\rho(x)}\nabla p(x,t)
	\right)
	=
	S(x,t)
$$

$$
	\begin{align*}
		\text{Source:}&&  p(x_0,t)\\
		\text{Boundaray:}&&  p(x,t)=0\\
		\text{Data:}&&  p(x_n,t)& \text{ for } n=1, \ldots, N
	\end{align*}
$$

**Inverse problem**:
Find acceptable $\kappa(x)$ and $\rho(x)$ such the wave equation satisfies initial and boundary conditions and reproduces the data. This could answer questions suchs as:
*How does sound in a concert hall bounce off the walls?* or 
*How does the internal body look (ultrasound scanning)?*



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


# Header 3

Inserting the prior on a plot (d versus m) and then inserting the function $d=g(m)$, the insertion of the function and the prior yields the posterior.


the posterior probability density of $\sigma(m)$:
$$
	\sigma(m) = \rho_m(m) L(m)
$$
where, $L(m) = \rho_d(g(m))$


If we have a high dimenisonal probability manifold, we may sample this manifold using Monte-Carlo.


# Ex 1
### Excercise
Determine density variations near the earth's surface, from a series of measurements of the horizontal component of the gravitational gradient at the surface.


# Ex 2
Density at each measuring location is described by;
$$
	d_j = \frac{∂g}{∂x} (x_j) =∫^∞_0 \frac{2G_\text{const.}z}{x_j^2 + z^2} ∆ρ(z) dz.
$$

The first step in solving this inverse problem is discretizing the integral. An initial idea is to replace the integration with a summation;
$$
    d_j = \sum_i^n \frac{2G_\text{const.}z_i}{x_j^2 + z_i^2} ∆ρ(z_i),
$$
but such a method yields no progress. Insted we do the integration manually:
$$
	\begin{align*}
	    d_j^i = G_\text{const.}\log
	        \left(
	            \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
	        \right)
	        \delta\rho_i
	    &&\Rightarrow&&
	    d_j = \sum_i G_\text{const.}\log
	        \left(
	            \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
	        \right)
	        \delta\rho_i
	\end{align*}
$$
# Ex 3
in which the summed-over term (excl. density variation) is;
$$
	\begin{align*}
	    G_{j,i} = G_\text{const.}\log
	        \left(
	            \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
	        \right).
	\end{align*}
$$
By calculating $G$ for each of our $x$-positions as well as a range of depths, $z$, we obtain the plot shown on the right.


# Ex 4
Now we have G, we can move on to the next step: **determine the parameters!**
$$
    \bar{m} = [G^TG + \epsilon^2I]^{-1} G^T d_\text{obs}
$$
this comes from minimizing: some regularized loss function, (see image on phone). The epsilon must be estimated, so lets just try a range, and plot an imshow of the stacked $\bar{m}$ vectors.


# Ex 5
##### Final mimization
Something something about what we are finding on the right...



