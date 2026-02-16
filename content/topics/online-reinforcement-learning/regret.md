# Regret Analysis


REGRET

KEY: intro
We introduce regret as a measure of performance. Regret describes the difference between the best possible reward and the reward actually received. This is helpful as unlike cumulative reward, regret does not grow indefinitely.

KEY: main
To evaluate regret we establish a baseline. In the stateless setting: we have a loss matrix which looks like the following:
$$
\begin{align}
\mathcal{L} =
\begin{bmatrix}
    l_{1,1} & l_{2,1} & \ldots & l_{t,1}\\
    l_{1,2} & l_{2,2} & \ldots & l_{t,2}\\
    \vdots & \vdots & \vdots & \vdots \\
    l_{1,|\mathcal{K}|} & l_{2,|\mathcal{K}|} & \ldots & l_{t,|\mathcal{K}|}\\
\end{bmatrix}
\end{align}
$$

This lets us define the regret as follows:
$$
\begin{align}
    R_t = \sum_{i=1}^t \min_{a \in \mathcal{K}} l_{t,a} - l_{t,\pi_t}
\end{align}
$$
The above equation describes the regret as the loss of the algorithm minus the best action applied contiously, i.e., the best row of the loss matrix in hindsight.

If the regret is of order $T$ $\Rightarrow$ no learning. We want sublinear regret, which means we are learning something.




#### Expected regret
$$
    \mathbb{E}[R_T] = \mathbb{E}[\sum_{t=1}^T l_{t, A_t}] - \mathbb{E}[\min_a \sum_{t=1}^T l_{t,a}]
$$


##### Oblivious adversary
$l_{t,a}$ are independent of actions. Meaning the adversary is not able to know our actions.

We will only consider oblivious adversary, in which case the last term in the equation above becomes deterministic.

##### Adaptive adversary
$l_{t,a}$ may depend on actions.


#### Pseudo regret
only defined in i.i.d. setting

$$
    \bar{R}_T = \mathbb{E}[\sum l_{t, A_t}] - min_a \mathbb{E}[\sum l_{t,a}]
$$
notice, here we have the minimum of the expectation rather than the expectation of the minimum.
$$
\begin{align*}
    \bar{R}_T &= \mathbb{E}[\sum l_{t, A_t}] - min_a \mu(a)T\\
        &= \mathbb{E}[\sum l_{t, A_t} - \mu^*]\\
        &= \mathbb{E}[\sum_t^T \Delta(A_t)]\\
        &= \sum_a \Delta(a) \mathbb{E}[N_T(a)]
\end{align*}
$$
where $\mu^* = \min_a \mu(a)$.

We also define $\Delta(a) = \mu(a)- \min \mu(a) = \mu(a) - \mu^*$


Notice, the pseudo regret is always less than or equal to the expected regret. We always use pseudo-regret for i.i.d. setting. The reason for this is that the comparator is more reasonable. The comparator we are using is $T\mu^*$...

KEY: example

Now we shall consider the iid bandit case (remember the bandit case is one where feedback is limited).

#### Exploration-exploitation trade-off
* action space width = 2
* T = known total time
* Delta is known

The actions yield $1/2 \pm \Delta$ respectively. If $\Delta$ is small, we should explore for a while, before determining which action is best. And then after that just repeat that action. 


We may bound the pseudo regret by:
$$
    \bar{R}_T \leq \frac{1}{2}\epsilon T\Delta + \delta(\epsilon) \Delta (1-\epsilon)T \leq (\frac{1}{2}\epsilon + \delta(\epsilon))\Delta T
$$
In which $\delta(\epsilon)$ is the probability that we selected the suboptimal action. We can bound this by:
$$ 
    \leq \mathbb{P} (\hat{\mu_{\epsilon T}} (a) \leq \hat{\mu_{\epsilon T}} (a^*))
$$ 
We can bound this chance of having chosen a bad arm by:
$$
    \leq \mathbb{P} (\hat{\mu_{\epsilon T}} (a^*) \geq 
    \hat{\mu_{\epsilon T}} (a^*) + \frac{1}{2}\Delta)
    + \mathbb{P}(\hat{\mu_{\epsilon T}} (a) \leq \mu(a) - \frac{1}{2}\Delta)
$$
We may employ Hoeffding's inequality to bound the first term:
$$
    \leq 2 e^{- \epsilon T \Delta^2 / 4}
$$
So now we can choose an epsilon to minimize the probability of choosing the bad arm.
$$
    \epsilon^* = \frac{4\ln (T\Delta^2)}{T\Delta^2}
$$
yielding a pseudo regret of:
$$
    \bar{R}_T \leq \frac{2(\ln(T\Delta^2)+1)}{\Delta}
$$

To summarize the above, it takes a longer time to dicern which action is better if the actions are closer together. It goes with $\Delta^2$. Each time we choose the wrong action, we lose $\Delta$, so pseudo regret goes with $\Delta$.