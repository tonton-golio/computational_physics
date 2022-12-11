
# intro
Linear tomography is a mathematical technique used to reconstruct an image from a set of projection data. In this case, we apply this technique to determine a map of density variations under the earth's surface using data from a series of seismographs that record the arrival times of signals from two earthquakes. Denser regions will cause faster wave propagation, resulting in an anomaly in the arrival time.

##### Expressing the time anomaly
The time anomaly $t_\gamma$ corresponding to a single ray can be expressed by the integral of the relative slowness along the ray:
$$
t_\gamma = \int_\gamma s(u) du
$$

This integral can be approximated using a discrete sum, as follows:
$$
t_\gamma = \sum_{i=1}^{N-1} s(u_i) (u_{i+1} - u_i)
$$
where $s(u)$ is the relative slowness at position $u$, $N$ is the total number of points along the ray, and $u_i$ is the position of the $i$-th point along the ray.

##### Defining the inverse problem
The matrix $G$ contains the paths traced by rays for each detector. If we let the slowness anomaly of each square of earth be the vector $m$, we can use $G$ to element-wise pick each passed square. This results in a linear system, so the expected observations are given by the matrix product:
$$d_{obs} = Gm$$


To see what the data looks like, I have made a demo:

# Generating data
To see what the data looks like, we can generate some sample data. We will constrain ourselves to a square setup with side length N and number of seismographs $n = N - 2$. This causes rays to hug the diagonal of an entered cell, so we don't need finer spacing on our rays. 

The constructed matrix $G$, summed over detectors and earthquakes and reshaped, allows us to verify that we have set up the matrix appropriately. The colors indicate how many rays enter each square. We add noise to the product of $G$ and m for more realistic data.

# G code
def make_G(N=13):
    G_right = [np.eye(N,k=1+i).flatten() for i in range(N-2)]
    G_left = [np.flip(np.eye(N,k=-(1+i)), axis=0).flatten() for i in range(N-2)]
    
    z = np.zeros((1,N**2))
    G = np.concatenate([z, G_left[::-1],z, z, G_right,z])

    G *= 2**.5 * 1000
    return G


# Predicting m
We have an underdetermined problem with 20 samples and 143 free parameters, yielding multiple solutions. However, significant noise makes it impossible for any solution to fit exactly, making it a mixed-determined problem. This makes the problem ill-posed and we apply Tikhonov regularization to predict $m$.

$$
\begin{align*}
    \bar{m} &= [G^TG + \epsilon^2\mathbf{I}]^{-1} G^T d_\text{obs}\\ 
    \rightarrow & ||\mathbf{d}_\text{obs} - \mathbf{G\hat{m}}||^2\approx N\sigma^2.
\end{align*}
$$
We move all terms to one side, and minimize as function of $\epsilon$.


# A delta function
##### A delta function
When applying our method to a true parameter vector shaped like a delta function, we obtain result great results if the square is traced by rays from both earthquakes, else we obtain large uncertainty along the single traced ray. The difference between the predicted $m$ and the true $m$ is the uncertainty. To fix this, we should try to minimize the number of entries in $m$.