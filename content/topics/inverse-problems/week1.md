
# Ex 1
### Density variations (Tikonov)
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



