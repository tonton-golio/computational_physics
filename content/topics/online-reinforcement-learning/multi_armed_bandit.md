KEY: intro
In this section we discuss multi armed bandits as the simplest example of reinforcement learning. We discuss methods for balancing exploration and exploitation. To asses the performance of an algorithm, we employ **regret**: the difference between the best possible reward and the reward actually received.

We will use the following notation: 
* $s$ is the state, 
* $a$ is the action, 
* $\mu$ is the mean reward, 
* $n(s,a)$ is the number of times the arm has been played, and, 
* $t$ is the time.


KEY: greedy policy
#### Greedy policy
We start by playing each arm once. Futher choices are based on entirely on mean reward so far. The problem with this is that we dont explore sufficiently.

KEY: epsilon greedy policy
#### $\epsilon$-greedy policy
The $\epsilon$-greedy policy includes a random component. It is a greedy policy only exploits, but we select an arm at random with probability $\epsilon$.
$$
\begin{align}
    \pi(a|s) = \begin{cases}
    \max{\mu} \quad\text{if } \text{random} > \epsilon \\
    \text{random} \quad\text{otherwise}
\end{cases}
\end{align}
$$


KEY: upper confidence bound
#### Upper Confidence Bound (UCB)
The UCB policy is a greedy policy that is biased towards exploration. But it takes into account how many times an arm has been played. 
$$
\begin{align}
\text{UCB}_t = \frac{\mu}{n(s,a)} + \sqrt{\frac{2\log t}{n(s,a)}}
\end{align}
$$
The best action is the one with the highest UCB. 

KEY: lower confidence bound
#### Lower Confidence Bound (LCB)
The lower confidence bound is the opposite of the upper confidence bound. It is a greedy policy that is biased towards exploitation. But it takes into account how many times an arm has been played.
$$
\begin{align}
    \text{LCB}_t = \hat{\mu_{t-1}}(a) - \sqrt{\frac{3\ln t}{2 N_{t-1}(a)}}
\end{align}
$$
The best action is the one with the lowest LCB.


KEY: experiment
### Experiment
We compare the performance of the four policies. The problem at hand is a four-armed bandit which give binomial rewards $\{0, 1\}$, with bias (win-probabilities) $\{0.1, 0.15, 0.2, 0.5\}$.




KEY: links
* [link to youtube vid](https://www.youtube.com/watch?v=e3L4VocZnnQ)
